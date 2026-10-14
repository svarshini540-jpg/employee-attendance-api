
"""
Employee Attendance & Analytics API
FastAPI + MongoDB

Run:
    python -m uvicorn app.main:app --reload

Environment variables:
    MONGO_URI
    MONGO_DB
"""

import os
import re
import json
import calendar
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Literal

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo import MongoClient, ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(override=False)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

client = MongoClient(
    MONGO_URI or "mongodb://localhost:27017",
    serverSelectionTimeoutMS=5000,
)

db = client[MONGO_DB]
employees = db["employees"]
attendance = db["attendance_logs"]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

PRESENCE_STATUSES = {"PRESENT", "WFH", "ON_DUTY"}
ALL_STATUSES = {"PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"}

app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Startup and indexes
# ---------------------------------------------------------------------------

@app.on_event("startup")
def create_indexes():
    employees.create_index(
        [("emp_code", ASCENDING)],
        unique=True,
        name="uq_employee_code",
    )
    employees.create_index(
        [("department", ASCENDING), ("joined_on", ASCENDING)],
        name="ix_employee_department_joined",
    )
    employees.create_index(
        [("joined_on", ASCENDING)],
        name="ix_employee_joined",
    )

    attendance.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="uq_attendance_employee_date",
    )
    attendance.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)],
        name="ix_attendance_date_employee",
    )
    attendance.create_index(
        [("emp_code", ASCENDING), ("punch_in", DESCENDING)],
        name="ix_attendance_employee_punchin",
    )
    attendance.create_index(
        [("status", ASCENDING), ("date", DESCENDING)],
        name="ix_attendance_status_date",
    )
    attendance.create_index(
        [("date", ASCENDING), ("status", ASCENDING)],
        name="ix_attendance_date_status",
    )


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def fail(status_code: int, detail: str):
    raise HTTPException(status_code=status_code, detail=detail)


def round_half_up(value, places=2):
    quantum = Decimal("1").scaleb(-places)
    return float(
        Decimal(str(value)).quantize(
            quantum,
            rounding=ROUND_HALF_UP,
        )
    )


def make_json_safe(value):
    """Convert BSON values, including binary data and timestamps, to JSON."""
    return json.loads(json_util.dumps(value))


def to_db_datetime(value: datetime) -> datetime:
    """Store UTC datetimes as naive BSON datetimes, truncated to seconds."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    value = value.astimezone(UTC).replace(microsecond=0)
    return value.replace(tzinfo=None)


def from_db_datetime(value: Optional[datetime]) -> Optional[datetime]:
    """Interpret MongoDB's naive BSON datetime as UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0)


def epoch_ms(value: Optional[datetime]) -> Optional[int]:
    value = from_db_datetime(value)
    if value is None:
        return None
    return int(value.timestamp() * 1000)


def parse_epoch_ms(value: Optional[int]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        fail(422, "Timestamps must be integer epoch milliseconds")
    if value < 100_000_000_000 or value > 4_102_444_800_000:
        fail(422, "Timestamp is outside the supported epoch-millisecond range")
    try:
        return to_db_datetime(datetime.fromtimestamp(value / 1000, UTC))
    except (ValueError, OverflowError, OSError):
        fail(422, "Invalid epoch-millisecond timestamp")


def parse_date(value: str, field_name="date") -> date:
    try:
        parsed = date.fromisoformat(value)
    except (ValueError, TypeError):
        fail(422, f"{field_name} must use YYYY-MM-DD format")
    if parsed.isoformat() != value:
        fail(422, f"{field_name} must use YYYY-MM-DD format")
    return parsed


def parse_month(value: str) -> tuple[date, date]:
    if not re.fullmatch(r"\d{4}-\d{2}", value or ""):
        fail(422, "month must use YYYY-MM format")
    try:
        year, month = map(int, value.split("-"))
        first = date(year, month, 1)
    except ValueError:
        fail(422, "Invalid month")
    last = date(year, month, calendar.monthrange(year, month)[1])
    return first, last


def parse_shift(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value or ""):
        fail(422, "Shift times must use HH:MM format")
    hour, minute = map(int, value.split(":"))
    return hour, minute


def ist_datetime(value: datetime) -> datetime:
    return from_db_datetime(value).astimezone(IST)


def is_weekday(day: date) -> bool:
    return day.weekday() < 5


def working_days_between(start: date, end: date) -> int:
    if end < start:
        return 0
    count = 0
    current = start
    while current <= end:
        if is_weekday(current):
            count += 1
        current += timedelta(days=1)
    return count


def validate_emp_code(value: str):
    if not re.fullmatch(r"EMP\d{4,6}", value or ""):
        fail(422, "emp_code must be EMP followed by 4 to 6 digits")
    return value


def get_employee(emp_code: str):
    employee = employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if not employee:
        fail(404, "Employee not found")
    return employee


def attendance_date_for(punch_in: datetime, employee: dict) -> date:
    """Determine the attendance date in IST, including overnight shifts."""
    local = ist_datetime(punch_in)
    start_h, start_m = parse_shift(employee["shift_start"])
    end_h, end_m = parse_shift(employee["shift_end"])
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m
    local_minutes = local.hour * 60 + local.minute

    if end_minutes <= start_minutes and local_minutes < end_minutes:
        return local.date() - timedelta(days=1)
    return local.date()


def shift_start_datetime(day: date, employee: dict) -> datetime:
    hour, minute = parse_shift(employee["shift_start"])
    return datetime.combine(day, time(hour, minute), tzinfo=IST)


def shift_end_datetime(day: date, employee: dict) -> datetime:
    start_h, start_m = parse_shift(employee["shift_start"])
    end_h, end_m = parse_shift(employee["shift_end"])
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m
    end_day = (
        day + timedelta(days=1)
        if end_minutes <= start_minutes
        else day
    )
    return datetime.combine(end_day, time(end_h, end_m), tzinfo=IST)


def compute_late_minutes(punch_in: datetime, day: date, employee: dict) -> int:
    actual = ist_datetime(punch_in)
    scheduled = shift_start_datetime(day, employee)
    seconds_late = int((actual - scheduled).total_seconds())
    if seconds_late > 600:
        return max(0, seconds_late // 60)
    return 0


def compute_overtime(punch_out: datetime, day: date, employee: dict) -> int:
    actual = ist_datetime(punch_out)
    scheduled_end = shift_end_datetime(day, employee)
    minutes = int((actual - scheduled_end).total_seconds() // 60)
    return max(0, minutes) if minutes >= 30 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    seconds = (punch_out - punch_in).total_seconds()
    return round_half_up(seconds / 3600, 2)


def derive_fields(doc: dict, employee: dict) -> dict:
    """Recalculate derived attendance values."""
    status = doc["status"]
    pin = doc.get("punch_in")
    pout = doc.get("punch_out")
    day = parse_date(doc["date"])

    if status not in PRESENCE_STATUSES or pin is None or pout is None:
        doc["work_hours"] = None
        doc["late_minutes"] = 0
        doc["overtime_minutes"] = 0
        doc["half_day"] = False
        return doc

    doc["work_hours"] = compute_work_hours(pin, pout)
    doc["late_minutes"] = compute_late_minutes(pin, day, employee)
    doc["overtime_minutes"] = compute_overtime(pout, day, employee)
    doc["half_day"] = doc["work_hours"] < 4.50
    return doc


def serialize_history(history: list) -> list:
    result = []
    for entry in history or []:
        item = dict(entry)
        item["at"] = epoch_ms(item.get("at"))
        changes = {}
        for key, value in (item.get("changes") or {}).items():
            if key in {"punch_in", "punch_out"}:
                changes[key] = epoch_ms(value) if value is not None else None
            else:
                changes[key] = value
        item["changes"] = changes
        result.append(item)
    return result


def serialize_attendance(doc: dict) -> dict:
    result = dict(doc)
    result.pop("_id", None)
    result["punch_in"] = epoch_ms(result.get("punch_in"))
    result["punch_out"] = epoch_ms(result.get("punch_out"))
    if "history" in result:
        result["history"] = serialize_history(result["history"])
    return result


def pagination(page: int, page_size: int):
    if page < 1:
        fail(422, "page must be at least 1")
    if page_size < 1 or page_size > 100:
        fail(422, "page_size must be between 1 and 100")
    return (page - 1) * page_size


def valid_status(status: str):
    if status not in ALL_STATUSES:
        fail(422, "Invalid attendance status")


def presence_status(status: str):
    if status not in PRESENCE_STATUSES:
        fail(422, "Status must be PRESENT, WFH, or ON_DUTY for punch operations")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emp_code: str
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=254)
    department: str = Field(min_length=1, max_length=100)
    shift_start: str = "09:30"
    shift_end: str = "18:30"
    joined_on: str

    @field_validator("emp_code")
    @classmethod
    def check_emp_code(cls, value):
        return validate_emp_code(value)

    @field_validator("shift_start", "shift_end")
    @classmethod
    def check_shift(cls, value):
        parse_shift(value)
        return value

    @field_validator("joined_on")
    @classmethod
    def check_joined_on(cls, value):
        parse_date(value, "joined_on")
        return value

    @field_validator("email")
    @classmethod
    def check_email(cls, value):
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            fail(422, "Invalid email address")
        return value


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emp_code: str
    punched_at: Optional[int] = None
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"

    @field_validator("emp_code")
    @classmethod
    def check_emp_code(cls, value):
        return validate_emp_code(value)

    @field_validator("punched_at")
    @classmethod
    def check_timestamp(cls, value):
        if value is not None:
            parse_epoch_ms(value)
        return value


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emp_code: str
    punched_at: Optional[int] = None

    @field_validator("emp_code")
    @classmethod
    def check_emp_code(cls, value):
        return validate_emp_code(value)

    @field_validator("punched_at")
    @classmethod
    def check_timestamp(cls, value):
        if value is not None:
            parse_epoch_ms(value)
        return value


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)
    status: Optional[str] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None

    @field_validator("status")
    @classmethod
    def check_status(cls, value):
        if value is not None:
            valid_status(value)
        return value

    @field_validator("punch_in", "punch_out")
    @classmethod
    def check_timestamp(cls, value):
        if value is not None:
            parse_epoch_ms(value)
        return value


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except PyMongoError:
        fail(503, "MongoDB is unavailable")


# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------

@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = to_db_datetime(datetime.now(UTC))
    try:
        employees.insert_one(doc)
    except DuplicateKeyError:
        fail(409, "emp_code already exists")
    except PyMongoError:
        fail(503, "Database operation failed")
    doc.pop("_id", None)
    doc["created_at"] = epoch_ms(doc["created_at"])
    return doc


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    skip = pagination(page, page_size)
    query = {}
    if department:
        query["department"] = department

    total = employees.count_documents(query)
    items = list(
        employees.find(query, {"_id": 0})
        .sort("emp_code", ASCENDING)
        .skip(skip)
        .limit(page_size)
    )
    for item in items:
        if isinstance(item.get("created_at"), datetime):
            item["created_at"] = epoch_ms(item["created_at"])
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
# Punch in / punch out
# ---------------------------------------------------------------------------

@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    employee = get_employee(body.emp_code)
    ts = (
        parse_epoch_ms(body.punched_at)
        if body.punched_at is not None
        else to_db_datetime(datetime.now(UTC))
    )
    day = attendance_date_for(ts, employee)
    doc = {
        "emp_code": body.emp_code,
        "date": day.isoformat(),
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(ts, day, employee),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        result = attendance.insert_one(doc)
    except DuplicateKeyError:
        fail(409, "Attendance already exists for this employee and date")
    except PyMongoError:
        fail(503, "Database operation failed")
    doc["_id"] = result.inserted_id
    return serialize_attendance(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    employee = get_employee(body.emp_code)
    ts = (
        parse_epoch_ms(body.punched_at)
        if body.punched_at is not None
        else to_db_datetime(datetime.now(UTC))
    )

    record = attendance.find_one(
        {
            "emp_code": body.emp_code,
            "punch_in": {"$lte": ts},
            "punch_out": None,
            "status": {"$in": list(PRESENCE_STATUSES)},
        },
        sort=[("punch_in", DESCENDING)],
    )

    if not record:
        existing = attendance.find_one(
            {"emp_code": body.emp_code, "punch_in": {"$lte": ts}},
            sort=[("punch_in", DESCENDING)],
        )
        if existing and existing.get("punch_out") is not None:
            fail(409, "Attendance record has already been punched out")
        fail(404, "No open punch-in record found")

    elapsed = (ts - record["punch_in"]).total_seconds()
    if elapsed <= 0:
        fail(422, "punch-out must be after punch-in")
    if elapsed > 24 * 60 * 60:
        fail(422, "punch-out cannot be more than 24 hours after punch-in")

    updated = attendance.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {"$set": {"punch_out": ts}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        fail(409, "Attendance record has already been punched out")

    updated = derive_fields(updated, employee)
    attendance.update_one(
        {"_id": updated["_id"]},
        {
            "$set": {
                "work_hours": updated["work_hours"],
                "late_minutes": updated["late_minutes"],
                "overtime_minutes": updated["overtime_minutes"],
                "half_day": updated["half_day"],
            }
        },
    )
    return serialize_attendance(updated)


# ---------------------------------------------------------------------------
# Attendance listing
# ---------------------------------------------------------------------------

@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    skip = pagination(page, page_size)
    query = {}

    if emp_code:
        validate_emp_code(emp_code)
        query["emp_code"] = emp_code
    if date_from:
        parse_date(date_from, "date_from")
    if date_to:
        parse_date(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        fail(422, "date_from must be on or before date_to")

    if date_from or date_to:
        query["date"] = {}
        if date_from:
            query["date"]["$gte"] = date_from
        if date_to:
            query["date"]["$lte"] = date_to

    if status:
        valid_status(status)
        query["status"] = status

    total = attendance.count_documents(query)
    docs = list(
        attendance.find(query)
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )
    return {
        "items": [serialize_attendance(doc) for doc in docs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
# Attendance regularization
# ---------------------------------------------------------------------------

@app.patch("/attendance/{emp_code}/{day}")
def regularize_attendance(emp_code: str, day: str, body: RegularizeIn):
    validate_emp_code(emp_code)
    parse_date(day, "date")
    employee = get_employee(emp_code)
    record = attendance.find_one({"emp_code": emp_code, "date": day})
    if not record:
        fail(404, "Attendance record not found")

    proposed = dict(record)
    requested = body.model_dump(exclude_unset=True)
    requested.pop("reason", None)
    requested.pop("regularized_by", None)

    for key in ("status", "punch_in", "punch_out"):
        if key in requested:
            proposed[key] = (
                parse_epoch_ms(requested[key])
                if key in {"punch_in", "punch_out"}
                and requested[key] is not None
                else requested[key]
            )

    status = proposed.get("status", record["status"])
    valid_status(status)

    if status in {"ABSENT", "LEAVE"}:
        if (
            ("punch_in" in requested and requested["punch_in"] is not None)
            or ("punch_out" in requested and requested["punch_out"] is not None)
        ):
            fail(422, "ABSENT and LEAVE records cannot have punch times")
        proposed["punch_in"] = None
        proposed["punch_out"] = None
    else:
        if proposed.get("punch_in") is None:
            fail(422, "Presence statuses require punch_in")
        pin = proposed["punch_in"]
        pout = proposed.get("punch_out")
        if attendance_date_for(pin, employee).isoformat() != day:
            fail(422, "punch_in must belong to the attendance date")
        if pout is not None:
            elapsed = (pout - pin).total_seconds()
            if elapsed <= 0:
                fail(422, "punch_out must be after punch_in")
            if elapsed > 24 * 60 * 60:
                fail(422, "punch_out cannot be more than 24 hours after punch-in")

    changed = {}
    for key in ("status", "punch_in", "punch_out"):
        if key not in requested:
            continue
        old_value = record.get(key)
        new_value = proposed.get(key)
        if old_value != new_value:
            changed[key] = new_value

    if status in {"ABSENT", "LEAVE"}:
        for key in ("punch_in", "punch_out"):
            if record.get(key) is not None:
                changed[key] = None

    if not changed:
        fail(422, "Regularization does not change any fields")

    proposed["status"] = status
    proposed["date"] = day
    proposed = derive_fields(proposed, employee)

    history_entry = {
        "at": to_db_datetime(datetime.now(UTC)),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changed,
    }

    old_history = record.get("history", [])
    update_filter = {
        "_id": record["_id"],
        "history": old_history,
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
    }
    set_values = {
        "status": proposed["status"],
        "punch_in": proposed.get("punch_in"),
        "punch_out": proposed.get("punch_out"),
        "work_hours": proposed.get("work_hours"),
        "late_minutes": proposed.get("late_minutes", 0),
        "overtime_minutes": proposed.get("overtime_minutes", 0),
        "half_day": proposed.get("half_day", False),
    }
    result = attendance.update_one(
        update_filter,
        {"$set": set_values, "$push": {"history": history_entry}},
    )
    if result.modified_count != 1:
        fail(409, "Attendance record changed concurrently; retry the request")

    updated = attendance.find_one({"_id": record["_id"]})
    return serialize_attendance(updated)


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def attendance_match_for_month(first: date, last: date, emp_code=None):
    query = {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}
    if emp_code:
        query["emp_code"] = emp_code
    return query


def attendance_month_aggregate(query: dict):
    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": {"emp_code": "$emp_code", "status": "$status"},
                "record_count": {"$sum": 1},
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                            1, 0,
                        ]
                    }
                },
                "late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "overtime_minutes": {
                    "$sum": {"$ifNull": ["$overtime_minutes", 0]}
                },
                "leave_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "on_duty_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}
                },
                "present_weekdays": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {
                                        "$lt": [
                                            {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                                            7,
                                        ]
                                    },
                                    {
                                        "$gt": [
                                            {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                                            1,
                                        ]
                                    },
                                ]
                            },
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]},
                            0,
                        ]
                    }
                },
                "work_hours_sum": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$ne": ["$work_hours", None]},
                                ]
                            },
                            "$work_hours", 0,
                        ]
                    }
                },
                "work_hours_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$ne": ["$work_hours", None]},
                                ]
                            },
                            1, 0,
                        ]
                    }
                },
            }
        },
    ]
    return list(attendance.aggregate(pipeline))


# ---------------------------------------------------------------------------
# Employee monthly analytics
# ---------------------------------------------------------------------------

@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str,
    month: str = Query(..., description="Month in YYYY-MM format"),
):
    employee = get_employee(validate_emp_code(emp_code))
    first, last = parse_month(month)
    joined = parse_date(employee["joined_on"], "joined_on")
    effective_start = max(first, joined)
    working_days = working_days_between(effective_start, last)

    groups = attendance_month_aggregate(
        attendance_match_for_month(first, last, emp_code)
    )
    present_days = 0.0
    leave_days = late_count = total_late = total_overtime = 0

    for group in groups:
        status = group["_id"]["status"]
        if status in PRESENCE_STATUSES:
            present_days += group.get("present_weekdays", 0)
        leave_days += group.get("leave_count", 0)
        late_count += group.get("late_count", 0)
        total_late += group.get("late_minutes", 0)
        total_overtime += group.get("overtime_minutes", 0)

    attendance_pct = (
        round_half_up(present_days / working_days * 100, 2)
        if working_days > 0 else None
    )
    return {
        "emp_code": emp_code,
        "name": employee["name"],
        "department": employee["department"],
        "month": month,
        "working_days": working_days,
        "present_days": round_half_up(present_days, 2),
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late,
        "total_overtime_minutes": total_overtime,
        "attendance_pct": attendance_pct,
    }


# ---------------------------------------------------------------------------
# Department monthly summary
# ---------------------------------------------------------------------------

@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., description="Month in YYYY-MM format"),
    department: Optional[str] = None,
):
    first, last = parse_month(month)
    employee_query = {"joined_on": {"$lte": last.isoformat()}}
    if department:
        employee_query["department"] = department

    active_employees = list(
        employees.find(
            employee_query,
            {"_id": 0, "emp_code": 1, "department": 1},
        )
    )
    if not active_employees:
        return {"month": month, "items": []}

    department_codes = {}
    for employee in active_employees:
        department_codes.setdefault(employee["department"], set()).add(
            employee["emp_code"]
        )

    query = {
        "date": {"$gte": first.isoformat(), "$lte": last.isoformat()},
        "emp_code": {"$in": [e["emp_code"] for e in active_employees]},
    }
    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": "$emp_code",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", list(PRESENCE_STATUSES)]},
                            1, 0,
                        ]
                    }
                },
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", list(PRESENCE_STATUSES)]},
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]},
                            0,
                        ]
                    }
                },
                "work_hours": {
                    "$avg": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$ne": ["$work_hours", None]},
                                ]
                            },
                            "$work_hours", None,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                            1, 0,
                        ]
                    }
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$late_minutes", 0]}
                },
                "leave_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "on_duty_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}
                },
            }
        },
    ]
    stats_by_code = {
        item["_id"]: item for item in attendance.aggregate(pipeline)
    }

    result = []
    for dept, codes in department_codes.items():
        stats = [stats_by_code[c] for c in codes if c in stats_by_code]
        present_days = sum(s.get("present_days", 0) for s in stats)
        hours_values = [
            s["work_hours"] for s in stats if s.get("work_hours") is not None
        ]
        result.append({
            "department": dept,
            "month": month,
            "headcount": len(codes),
            "present_days": round_half_up(present_days, 2),
            "avg_work_hours": (
                round_half_up(sum(hours_values) / len(hours_values), 2)
                if hours_values else None
            ),
            "late_count": sum(s.get("late_count", 0) for s in stats),
            "total_late_minutes": sum(
                s.get("total_late_minutes", 0) for s in stats
            ),
            "leave_count": sum(s.get("leave_count", 0) for s in stats),
            "on_duty_count": sum(s.get("on_duty_count", 0) for s in stats),
        })
    result.sort(key=lambda item: item["department"])
    return {"month": month, "items": result}


# ---------------------------------------------------------------------------
# Late leaderboard
# ---------------------------------------------------------------------------

@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., description="Month in YYYY-MM format"),
    limit: int = Query(10, ge=1, le=100),
    department: Optional[str] = None,
):
    first, last = parse_month(month)
    employee_query = {}
    if department:
        employee_query["department"] = department

    eligible = list(
        employees.find(
            employee_query,
            {"_id": 0, "emp_code": 1, "name": 1, "department": 1},
        )
    )
    employee_map = {e["emp_code"]: e for e in eligible}
    if not employee_map:
        return {"month": month, "items": []}

    pipeline = [
        {
            "$match": {
                "date": {"$gte": first.isoformat(), "$lte": last.isoformat()},
                "emp_code": {"$in": list(employee_map.keys())},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$late_minutes", 0]}
                },
            }
        },
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
    ]
    rows = list(attendance.aggregate(pipeline))
    ranked = []
    previous_total = None
    current_rank = 0

    for index, row in enumerate(rows, start=1):
        total = row["total_late_minutes"]
        if total != previous_total:
            current_rank = index
            previous_total = total
        employee = employee_map.get(row["_id"])
        if not employee:
            continue
        ranked.append({
            "rank": current_rank,
            "emp_code": row["_id"],
            "name": employee["name"],
            "department": employee["department"],
            "total_late_minutes": total,
        })

    if len(ranked) > limit:
        cutoff_rank = ranked[limit - 1]["rank"]
        ranked = [item for item in ranked if item["rank"] <= cutoff_rank]
    return {"month": month, "items": ranked}


# ---------------------------------------------------------------------------
# Department daily trend
# ---------------------------------------------------------------------------

@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    date_from: str = Query(..., alias="from"),
    date_to: str = Query(..., alias="to"),
):
    start = parse_date(date_from, "from")
    end = parse_date(date_to, "to")
    if end < start:
        fail(422, "to must be on or after from")
    if (end - start).days + 1 > 92:
        fail(422, "Date range cannot exceed 92 days")

    dept_employees = list(
        employees.find(
            {"department": department},
            {"_id": 0, "emp_code": 1, "joined_on": 1},
        )
    )
    if not dept_employees:
        fail(404, "Department not found")

    codes = [item["emp_code"] for item in dept_employees]
    join_dates = {
        item["emp_code"]: parse_date(item["joined_on"], "joined_on")
        for item in dept_employees
    }

    pipeline = [
        {
            "$match": {
                "emp_code": {"$in": codes},
                "date": {"$gte": start.isoformat(), "$lte": end.isoformat()},
            }
        },
        {
            "$group": {
                "_id": "$date",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", list(PRESENCE_STATUSES)]},
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1]},
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                            1, 0,
                        ]
                    }
                },
            }
        },
        {"$sort": {"_id": 1}},
    ]
    stats = {item["_id"]: item for item in attendance.aggregate(pipeline)}
    result = []
    current = start
    previous_rates = []

    while current <= end:
        key = current.isoformat()
        row = stats.get(key, {})
        headcount = sum(1 for joined in join_dates.values() if joined <= current)
        present_count = round_half_up(row.get("present_count", 0), 2)
        late_count = row.get("late_count", 0)
        working_day = is_weekday(current)

        rate = (
            round_half_up(present_count / headcount, 4)
            if working_day and headcount > 0 else None
        )
        if rate is not None:
            previous_rates.append(rate)
        recent = previous_rates[-7:]
        moving_avg = (
            round_half_up(sum(recent) / len(recent), 4) if recent else None
        )
        result.append({
            "date": key,
            "is_working_day": working_day,
            "headcount": headcount,
            "present_count": present_count,
            "late_count": late_count,
            "attendance_rate": rate,
            "moving_avg_7d": moving_avg,
        })
        current += timedelta(days=1)

    return {
        "department": department,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "items": result,
    }


# ---------------------------------------------------------------------------
# Query explain support
# ---------------------------------------------------------------------------

EXPLAIN_ENDPOINTS = {
    "attendance_list",
    "employee_monthly",
    "department_summary",
    "late_leaderboard",
    "department_trend",
}


def run_explain(collection_name: str, pipeline: list):
    command = {
        "aggregate": collection_name,
        "pipeline": pipeline,
        "cursor": {},
    }
    return db.command("explain", command, verbosity="executionStats")


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    month: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    date_range_from: Optional[str] = Query(None, alias="from"),
    date_range_to: Optional[str] = Query(None, alias="to"),
):
    if endpoint not in EXPLAIN_ENDPOINTS:
        fail(422, "Unsupported explain endpoint")

    skip = pagination(page, page_size)
    collection_name = "attendance_logs"
    pipeline = []

    if endpoint == "attendance_list":
        query = {}
        if emp_code:
            validate_emp_code(emp_code)
            query["emp_code"] = emp_code
        if date_from:
            parse_date(date_from, "date_from")
        if date_to:
            parse_date(date_to, "date_to")
        if date_from and date_to and date_from > date_to:
            fail(422, "date_from must be on or before date_to")
        if date_from or date_to:
            query["date"] = {}
            if date_from:
                query["date"]["$gte"] = date_from
            if date_to:
                query["date"]["$lte"] = date_to
        if status:
            valid_status(status)
            query["status"] = status
        
        effective_limit = min(limit, page_size)

        pipeline = [
            {"$match": query},
            {"$sort": {"date": -1, "emp_code": 1}},
            {"$skip": skip},
            {"$limit": effective_limit},
        ]


    elif endpoint == "employee_monthly":
        if not emp_code or not month:
            fail(422, "employee_monthly requires emp_code and month")
        get_employee(validate_emp_code(emp_code))
        first, last = parse_month(month)
        pipeline = [
            {"$match": attendance_match_for_month(first, last, emp_code)},
            {
                "$group": {
                    "_id": "$emp_code",
                    "total_late_minutes": {
                        "$sum": {"$ifNull": ["$late_minutes", 0]}
                    },
                    "total_overtime_minutes": {
                        "$sum": {"$ifNull": ["$overtime_minutes", 0]}
                    },
                }
            },
        ]

    elif endpoint == "department_summary":
        if not month:
            fail(422, "department_summary requires month")
        first, last = parse_month(month)
        pipeline = [
            {
                "$match": {
                    "date": {"$gte": first.isoformat(), "$lte": last.isoformat()}
                }
            },
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "total_late_minutes": {
                        "$sum": {"$ifNull": ["$late_minutes", 0]}
                    },
                }
            },
        ]

    elif endpoint == "late_leaderboard":
        if not month:
            fail(422, "late_leaderboard requires month")
        first, last = parse_month(month)
        employee_query = {}
        if department:
            employee_query["department"] = department
        employee_codes = [
            item["emp_code"]
            for item in employees.find(employee_query, {"_id": 0, "emp_code": 1})
        ]
        pipeline = [
            {
                "$match": {
                    "date": {"$gte": first.isoformat(), "$lte": last.isoformat()},
                    "emp_code": {"$in": employee_codes},
                }
            },
            {
                "$group": {
                    "_id": "$emp_code",
                    "total_late_minutes": {
                        "$sum": {"$ifNull": ["$late_minutes", 0]}
                    },
                }
            },
            {"$match": {"total_late_minutes": {"$gt": 0}}},
            {"$sort": {"total_late_minutes": -1, "_id": 1}},
            {"$limit": limit},
        ]

    elif endpoint == "department_trend":
        if not department or not date_range_from or not date_range_to:
            fail(422, "department_trend requires department, from, and to")
        start = parse_date(date_range_from, "from")
        end = parse_date(date_range_to, "to")
        if end < start or (end - start).days + 1 > 92:
            fail(422, "Invalid trend date range")

        dept_employees = list(
            employees.find({"department": department}, {"_id": 0, "emp_code": 1})
        )
        if not dept_employees:
            fail(404, "Department not found")

        pipeline = [
            {
                "$match": {
                    "emp_code": {
                        "$in": [item["emp_code"] for item in dept_employees]
                    },
                    "date": {"$gte": start.isoformat(), "$lte": end.isoformat()},
                }
            },
            {
                "$group": {
                    "_id": "$date",
                    "present_count": {
                        "$sum": {
                            "$cond": [
                                {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                1, 0,
                            ]
                        }
                    },
                }
            },
            {"$sort": {"_id": 1}},
        ]

    try:
        explanation = run_explain(collection_name, pipeline)
    except PyMongoError as exc:
        fail(503, f"Could not explain query: {exc.__class__.__name__}")

    return {
        "endpoint": endpoint,
        "collection": collection_name,
        # BSON Extended JSON safely handles binary values and timestamps.
        "explain": make_json_safe(explanation),
    }
