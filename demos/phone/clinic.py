"""The clinic's booking database: the only truth about appointments. The agent changes it through tools, so a
booking, move or cancellation is a real database write, not something the model remembers.

SQLite (stdlib), one file, survives restarts. Nothing about an existing booking is told or changed unless the
caller's name AND phone number match it. Every change is written to an audit table. To connect a real
practice-management system, implement the same six methods (open_slots, read_back_phone, find_booking, book,
move, cancel) against its API.

    CLINIC_DB=data/clinic.sqlite        database file (created and seeded with demo patients on first use)
    CLINIC_COUNTRY_CODE=44              country code for local numbers ("07700 ..." -> +447700...)
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time

SLOTS = ["Monday 09:00", "Monday 16:00", "Tuesday 10:00", "Tuesday 15:30", "Wednesday 09:00",
         "Wednesday 11:00", "Thursday 14:00", "Friday 11:30"]
SEED = [("Monday 09:00", "Maria Lopez", "+447700900111"),  # fictional demo patients
        ("Monday 16:00", "Tom Reed", "+447700900222"),
        ("Wednesday 11:00", "Priya Shah", "+447700900333")]

_WORDS = {"zero": "0", "oh": "0", "o": "0", "nought": "0", "one": "1", "two": "2", "three": "3", "four": "4",
          "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_REPEAT = {"double": 2, "triple": 3}
_SAY = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def spoken_digits(phone: str) -> tuple[bool, str]:
    """Phone number as the caller said it ("plus four four, seven seven double oh, nine hundred" or "+44 7700 900")
    -> (said with a plus, digits). Done in code: a model once turned a correctly read-back number into the
    wrong digits."""
    s = (phone or "").casefold()
    plus = s.strip().startswith("+") or bool(re.match(r"\s*plus\b", s))
    out, repeat = [], 1
    for tok in re.findall(r"[a-z]+|\d", s):
        if tok in _REPEAT:
            repeat = _REPEAT[tok]
            continue
        d = tok if tok.isdigit() else _WORDS.get(tok)
        if d is not None:
            out.append(d * repeat)
        repeat = 1
    return plus, "".join(out)


def e164(phone: str, country: str | None = None) -> str:
    """One canonical form, so "+44 7700 900456", "0044 7700..." and "07700 900456" are the same patient."""
    country = country or os.getenv("CLINIC_COUNTRY_CODE", "44")
    plus, d = spoken_digits(phone)
    if not d:
        return ""
    if plus:
        return "+" + d
    if d.startswith("00"):
        return "+" + d[2:]
    if d.startswith("0"):
        return "+" + country + d[1:]
    return "+" + (d if len(d) > 10 else country + d)


def _name(name: str) -> str:
    return " ".join((name or "").casefold().split())


class Diary:
    def __init__(self, path: str | None = None, seed: bool = True):
        self.path = path or os.getenv("CLINIC_DB", "data/clinic.sqlite")
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS appointments (
                slot TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT NOT NULL, phone TEXT NOT NULL,
                created REAL NOT NULL, updated REAL NOT NULL, UNIQUE (name_key, phone));
            CREATE TABLE IF NOT EXISTS audit (ts REAL NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL);""")
        if seed and not self.db.execute("SELECT 1 FROM audit LIMIT 1").fetchone():
            for slot, name, phone in SEED:
                self._insert(slot, name, phone)
            self._audit("seed", f"{len(SEED)} demo patients")

    # --- storage -------------------------------------------------------------------------------------------
    def _insert(self, slot, name, phone):
        now = time.time()
        self.db.execute("INSERT INTO appointments VALUES (?, ?, ?, ?, ?, ?)",
                        (slot, name.strip(), _name(name), phone, now, now))

    def _audit(self, action, detail):
        self.db.execute("INSERT INTO audit VALUES (?, ?, ?)", (time.time(), action, detail))

    def _slot(self, slot: str) -> str | None:
        want = " ".join((slot or "").casefold().split())
        return next((s for s in SLOTS if s.casefold() == want), None)

    def _find(self, name: str, phone: str) -> str | None:
        n, p = _name(name), e164(phone)
        if not n or len(p) < 8:
            return None
        row = self.db.execute("SELECT slot FROM appointments WHERE name_key = ? AND phone = ?", (n, p)).fetchone()
        return row[0] if row else None

    @property
    def bookings(self) -> dict:
        return {s: {"name": n, "phone": p} for s, n, p in self.db.execute("SELECT slot, name, phone FROM appointments")}

    @property
    def log(self) -> list[str]:
        return [f"{a} {d}" for a, d in self.db.execute("SELECT action, detail FROM audit ORDER BY rowid") if a != "seed"]

    def audit_rows(self, limit: int = 50) -> list[tuple]:
        return self.db.execute("SELECT ts, action, detail FROM audit ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()

    # --- the six tool methods ---------------------------------------------------------------------------------
    def open_slots(self) -> dict:
        taken = {r[0] for r in self.db.execute("SELECT slot FROM appointments")}
        return {"open_slots": [s for s in SLOTS if s not in taken]}

    def read_back_phone(self, phone: str) -> dict:
        """The exact words to read a number back with, as the caller said it, so they confirm the digits."""
        plus, d = spoken_digits(phone)
        if len(d) < 7:
            return {"ok": False, "message": "That is not a full phone number; ask for it again."}
        cut = [len(d) - 6, len(d) - 3]  # e.g. 07700 900 456, 07700 900 123
        groups = [d[:cut[0]], d[cut[0]:cut[1]], d[cut[1]:]]
        said = ", ".join(" ".join(_SAY[int(c)] for c in g) for g in groups)
        return {"ok": True, "say": ("plus " if plus else "") + said}

    def find_booking(self, name: str, phone: str) -> dict:
        s = self._find(name, phone)
        if s is None:  # same answer whether the name, the number or both are wrong: no fishing for bookings
            return {"found": False, "message": "No booking matches that name and phone number."}
        return {"found": True, "slot": s, "name": self.bookings[s]["name"]}

    def book(self, name: str, phone: str, slot: str) -> dict:
        with self._lock:
            s, p = self._slot(slot), e164(phone)
            if s is None:
                return {"ok": False, "message": f"'{slot}' is not a slot.", **self.open_slots()}
            if not _name(name) or len(p) < 8:
                return {"ok": False, "message": "A full name and phone number are needed to book."}
            existing = self._find(name, phone)
            if existing:
                return {"ok": False, "message": "This patient already has a booking; move it instead.",
                        "existing_slot": existing}
            try:
                self._insert(s, name, p)
            except sqlite3.IntegrityError:
                return {"ok": False, "message": f"{s} is already taken.", **self.open_slots()}
            self._audit("booked", f"{s} for {name.strip()}")
            return {"ok": True, "slot": s}

    def move(self, name: str, phone: str, new_slot: str) -> dict:
        with self._lock:
            old = self._find(name, phone)
            if old is None:
                return {"ok": False, "message": "No booking matches that name and phone number; nothing was changed."}
            s = self._slot(new_slot)
            if s is None:
                return {"ok": False, "message": f"'{new_slot}' is not a slot.", "current_slot": old, **self.open_slots()}
            if s != old:
                try:
                    self.db.execute("UPDATE appointments SET slot = ?, updated = ? WHERE slot = ?", (s, time.time(), old))
                except sqlite3.IntegrityError:
                    return {"ok": False, "message": f"{s} is not available.", "current_slot": old, **self.open_slots()}
                self._audit("moved", f"{self.bookings[s]['name']} from {old} to {s}")
            return {"ok": True, "old_slot": old, "new_slot": s}

    def cancel(self, name: str, phone: str) -> dict:
        with self._lock:
            old = self._find(name, phone)
            if old is None:
                return {"ok": False, "message": "No booking matches that name and phone number; nothing was changed."}
            who = self.bookings[old]["name"]
            self.db.execute("DELETE FROM appointments WHERE slot = ?", (old,))
            self._audit("cancelled", f"{old} for {who}")
            return {"ok": True, "cancelled_slot": old}


if __name__ == "__main__":
    os.environ["CLINIC_COUNTRY_CODE"] = "44"
    assert spoken_digits("zero seven seven zero zero, nine zero zero, four five six") == (False, "07700900456")
    assert spoken_digits("oh seven seven double oh, 900 one two three") == (False, "07700900123")
    assert e164("plus four four seven seven zero zero nine zero zero four five six") == "+447700900456"
    assert e164("07700 900456") == e164("+44 7700 900456") == e164("00447700900456") == "+447700900456"
    d = Diary(":memory:")
    assert d.read_back_phone("+44 7700 900456")["say"] == "plus four four seven seven zero zero, nine zero zero, four five six"
    assert d.find_booking("Maria Lopez", "+44 7700 900 111")["slot"] == "Monday 09:00"
    assert not d.find_booking("Maria Lopez", "+44 7700 900 999")["found"]            # wrong number: nothing told
    assert not d.move("Someone Else", "07700900456", "Friday 11:30")["ok"]           # unverified: nothing changed
    assert d.book("Alex Morgan", "plus four four seven seven zero zero nine zero zero four five six", "friday 11:30")["ok"]
    assert d.find_booking("Alex Morgan", "07700 900 456")["slot"] == "Friday 11:30"  # local form finds it too
    assert not d.book("Sam Taylor", "07700900123", "Friday 11:30")["ok"]            # taken
    assert d.move("alex morgan", "07700900456", "Thursday 14:00") == {"ok": True, "old_slot": "Friday 11:30", "new_slot": "Thursday 14:00"}
    assert "Friday 11:30" in d.open_slots()["open_slots"]
    assert d.cancel("Alex Morgan", "07700900456")["ok"] and not d.find_booking("Alex Morgan", "07700900456")["found"]
    assert d.log == ["booked Friday 11:30 for Alex Morgan", "moved Alex Morgan from Friday 11:30 to Thursday 14:00",
                     "cancelled Thursday 14:00 for Alex Morgan"], d.log
    print("ok")
