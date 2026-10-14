"""
Employee Attendance & Analytics API
FastAPI + MongoDB

Run:
    uvicorn app.main:app --port 8000

Environment:
    MONGO_URI
    MONGO_DB
"""

import os
import re
import calendar
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Literal

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo import (
    MongoClient,
    ASCENDING,
    DESCENDING,
    ReturnDocument,
)
from pymongo.errors import DuplicateKeyError, PyMongoError


load_dotenv(override=False)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI must be set in the environment or .env file")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client[MONGO_DB]
employees = db["employees"]
attendance = db["attendance_logs"]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

PRESENCE = ["PRESENT", "WFH", "ON_DUTY"]
STATUSES = ["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]

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

def fail(code: int, detail: str):
    raise HTTPException(status_code=code, detail=detail)


def round_half_up(value, places=2):
    quantum = Decimal("1").scaleb(-places)
    return float(
        Decimal(str(value)).quantize(
            quantum,
            rounding=ROUND_HALF_UP,
        )
    )


def json_safe(value):
    import json
    return json.loads(json_util.dumps(value))


def to_db_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0, tzinfo=None)


def from_db_datetime(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0)


def epoch_ms(value):
    value = from_db_datetime(value)
    return int(value.timestamp() * 1000) if value else None


def parse_epoch_ms(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        fail(422, "Timestamp must be integer epoch milliseconds")
    if not 100_000_000_000 <= value <= 4_102_444_800_000:
        fail(422, "Timestamp outside supported epoch-millisecond range")
    try:
        return to_db_datetime(datetime.fromtimestamp(value / 1000, UTC))
    except (ValueError, OverflowError, OSError):
        fail(422, "Invalid timestamp")


def parse_date(value: str, field="date") -> date:
    try:
        result = date.fromisoformat(value)
    except (ValueError, TypeError):
        fail(422, f"{field} must use YYYY-MM-DD format")
    if result.isoformat() != value:
        fail(422, f"{field} must use YYYY-MM-DD format")
    return result


def parse_month(value: str):
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value or ""):
        fail(422, "month must use YYYY-MM format")
    year, month = map(int, value.split("-"))
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    return first, last


def parse_shift(value: str):
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value or ""):
        fail(422, "Shift times must use HH:MM format")
    hour, minute = map(int, value.split(":"))
    return hour, minute


def validate_emp_code(value: str):
    if not re.fullmatch(r"EMP\d{4,6}", value or ""):
        fail(422, "emp_code must be EMP followed by 4 to 6 digits")
    return value


def is_weekday(day: date):
    return day.weekday() < 5


def working_days_between(start: date, end: date):
    if end < start:
        return 0
    total = 0
    current = start
    while current <= end:
        total += int(is_weekday(current))
        current += timedelta(days=1)
    return total


def pagination(page, page_size):
    if page < 1 or not 1 <= page_size <= 100:
        fail(422, "Invalid pagination parameters")
    return (page - 1) * page_size


def get_employee(emp_code: str):
    employee = employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if not employee:
        fail(404, "Employee not found")
    return employee


def ist_datetime(value: datetime):
    return from_db_datetime(value).astimezone(IST)


def attendance_date_for(punch_in: datetime, employee: dict):
    local = ist_datetime(punch_in)
    sh, sm = parse_shift(employee["shift_start"])
    eh, em = parse_shift(employee["shift_end"])
    start = sh * 60 + sm
    end = eh * 60 + em
    current = local.hour * 60 + local.minute

    if end <= start and current < end:
        return local.date() - timedelta(days=1)
    return local.date()


def shift_start_datetime(day: date, employee: dict):
    hour, minute = parse_shift(employee["shift_start"])
    return datetime.combine(day, time(hour, minute), tzinfo=IST)


def shift_end_datetime(day: date, employee: dict):
    sh, sm = parse_shift(employee["shift_start"])
    eh, em = parse_shift(employee["shift_end"])
    next_day = day + timedelta(days=1) if (eh * 60 + em) <= (sh * 60 + sm) else day
    return datetime.combine(next_day, time(eh, em), tzinfo=IST)


def compute_late_minutes(punch_in, day, employee):
    seconds = int(
        (ist_datetime(punch_in) - shift_start_datetime(day, employee))
        .total_seconds()
    )
    return seconds // 60 if seconds > 600 else 0


def compute_overtime(punch_out, day, employee):
    seconds = (
        ist_datetime(punch_out) - shift_end_datetime(day, employee)
    ).total_seconds()
    minutes = int(seconds // 60)
    return minutes if minutes >= 30 else 0


def compute_work_hours(punch_in, punch_out):
    seconds = (punch_out - punch_in).total_seconds()
    return round_half_up(seconds / 3600, 2)


def derive_fields(doc, employee):
    status = doc["status"]
    pin = doc.get("punch_in")
    pout = doc.get("punch_out")

    if status not in PRESENCE or pin is None:
        doc.update(
            work_hours=None,
            late_minutes=0,
            overtime_minutes=0,
            half_day=False,
        )
        return doc

    day = parse_date(doc["date"])
    doc["late_minutes"] = compute_late_minutes(pin, day, employee)

    if pout is None:
        doc.update(work_hours=None, overtime_minutes=0, half_day=False)
        return doc

    doc["work_hours"] = compute_work_hours(pin, pout)
    doc["overtime_minutes"] = compute_overtime(pout, day, employee)
    doc["half_day"] = doc["work_hours"] < 4.50
    return doc


def serialize_history(history):
    output = []
    for entry in history or []:
        item = dict(entry)
        item["at"] = epoch_ms(item.get("at"))
        converted = {}
        for field, values in (item.get("changes") or {}).items():
            values = dict(values)
            if field in ("punch_in", "punch_out"):
                for key in ("from", "to"):
                    if values.get(key) is not None:
                        values[key] = epoch_ms(values[key])
            converted[field] = values
        item["changes"] = converted
        output.append(item)
    return output


def serialize_attendance(doc):
    result = dict(doc)
    result.pop("_id", None)
    result["punch_in"] = epoch_ms(result.get("punch_in"))
    result["punch_out"] = epoch_ms(result.get("punch_out"))
    result["history"] = serialize_history(result.get("history", []))
    return result


def validate_status(status):
    if status not in STATUSES:
        fail(422, "Invalid attendance status")


def month_match(first, last, emp_code=None):
    query = {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}
    if emp_code:
        query["emp_code"] = emp_code
    return query


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emp_code: str
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=120)
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = "09:30"
    shift_end: str = "18:30"
    joined_on: str

    @field_validator("emp_code")
    @classmethod
    def emp_code_valid(cls, value):
        return validate_emp_code(value)

    @field_validator("name", "department")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            fail(422, "Field must not be blank")
        return value

    @field_validator("email")
    @classmethod
    def email_valid(cls, value):
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            fail(422, "Invalid email address")
        return value

    @field_validator("shift_start", "shift_end")
    @classmethod
    def shift_valid(cls, value):
        parse_shift(value)
        return value

    @field_validator("shift_end")
    @classmethod
    def shift_differs(cls, value, info):
        if info.data.get("shift_start") == value:
            fail(422, "shift_start and shift_end must differ")
        return value

    @field_validator("joined_on")
    @classmethod
    def join_date_valid(cls, value):
        parse_date(value, "joined_on")
        return value


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emp_code: str
    punched_at: Optional[int] = None
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"

    @field_validator("emp_code")
    @classmethod
    def emp_code_valid(cls, value):
        return validate_emp_code(value)

    @field_validator("punched_at", mode="before")
    @classmethod
    def timestamp_valid(cls, value):
        if value is not None:
            parse_epoch_ms(value)
        return value


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emp_code: str
    punched_at: Optional[int] = None

    @field_validator("emp_code")
    @classmethod
    def emp_code_valid(cls, value):
        return validate_emp_code(value)

    @field_validator("punched_at", mode="before")
    @classmethod
    def timestamp_valid(cls, value):
        if value is not None:
            parse_epoch_ms(value)
        return value


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)
    status: Optional[Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None

    @field_validator("punch_in", "punch_out", mode="before")
    @classmethod
    def timestamp_valid(cls, value):
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
        fail(409, "Employee already exists")
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
    query = {"department": department} if department is not None else {}
    total = employees.count_documents(query)

    items = list(
        employees.find(query, {"_id": 0})
        .sort("emp_code", ASCENDING)
        .skip(skip)
        .limit(page_size)
    )

    for item in items:
        item["created_at"] = epoch_ms(item["created_at"])

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
# Punch operations
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
        attendance.insert_one(doc)
    except DuplicateKeyError:
        fail(409, "Attendance already exists for this employee and date")
    except PyMongoError:
        fail(503, "Database operation failed")

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
            "status": {"$in": PRESENCE},
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
    if elapsed > 86400:
        fail(422, "punch-out cannot be more than 24 hours after punch-in")

    calculated = derive_fields({**record, "punch_out": ts}, employee)

    updated = attendance.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": ts,
                "work_hours": calculated["work_hours"],
                "late_minutes": calculated["late_minutes"],
                "overtime_minutes": calculated["overtime_minutes"],
                "half_day": calculated["half_day"],
            }
        },
        return_document=ReturnDocument.AFTER,
    )

    if updated is None:
        fail(409, "Attendance record has already been punched out")

    return serialize_attendance(updated)


# ---------------------------------------------------------------------------
# Attendance listing
# ---------------------------------------------------------------------------

def attendance_list_query(emp_code, date_from, date_to, status):
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
        validate_status(status)
        query["status"] = status

    return query


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    query = attendance_list_query(emp_code, date_from, date_to, status)
    skip = pagination(page, page_size)

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
# Regularization and audit history
# ---------------------------------------------------------------------------

@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    validate_emp_code(emp_code)
    parse_date(date, "date")
    employee = get_employee(emp_code)

    record = attendance.find_one({"emp_code": emp_code, "date": date})
    if not record:
        fail(404, "Attendance record not found")

    requested = body.model_dump(exclude_unset=True)
    reason = requested.pop("reason")
    regularized_by = requested.pop("regularized_by")

    proposed = dict(record)
    for key in ("status", "punch_in", "punch_out"):
        if key in requested:
            proposed[key] = (
                parse_epoch_ms(requested[key])
                if key in ("punch_in", "punch_out") and requested[key] is not None
                else requested[key]
            )

    status = proposed["status"]
    validate_status(status)

    if status in ("ABSENT", "LEAVE"):
        if (
            ("punch_in" in requested and requested["punch_in"] is not None)
            or ("punch_out" in requested and requested["punch_out"] is not None)
        ):
            fail(422, "ABSENT and LEAVE cannot have punch times")
        proposed["punch_in"] = None
        proposed["punch_out"] = None
    else:
        if proposed.get("punch_in") is None:
            fail(422, "Presence statuses require punch_in")

        if attendance_date_for(proposed["punch_in"], employee).isoformat() != date:
            fail(422, "punch_in must belong to the attendance date")

        if proposed.get("punch_out") is not None:
            elapsed = (
                proposed["punch_out"] - proposed["punch_in"]
            ).total_seconds()
            if elapsed <= 0 or elapsed > 86400:
                fail(422, "punch_out must be after punch_in and within 24 hours")

    proposed["status"] = status
    proposed["date"] = date
    proposed = derive_fields(proposed, employee)

    # Audit actual changes, including changes to derived fields.
    fields = (
        "status", "punch_in", "punch_out",
        "work_hours", "late_minutes", "overtime_minutes", "half_day",
    )
    changes = {}

    for field in fields:
        old = record.get(field)
        new = proposed.get(field)
        if old != new:
            changes[field] = {"from": old, "to": new}

    if not changes:
        fail(422, "Regularization does not change any fields")

    history_entry = {
        "at": to_db_datetime(datetime.now(UTC)),
        "by": regularized_by,
        "reason": reason,
        "changes": changes,
    }

    # Compare-and-update prevents concurrent edits from overwriting history.
    update_filter = {
        "_id": record["_id"],
        "history": record.get("history", []),
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
    }

    result = attendance.update_one(
        update_filter,
        {
            "$set": {
                "status": proposed["status"],
                "punch_in": proposed.get("punch_in"),
                "punch_out": proposed.get("punch_out"),
                "work_hours": proposed.get("work_hours"),
                "late_minutes": proposed.get("late_minutes", 0),
                "overtime_minutes": proposed.get("overtime_minutes", 0),
                "half_day": proposed.get("half_day", False),
            },
            "$push": {"history": history_entry},
        },
    )

    if result.modified_count != 1:
        fail(409, "Attendance record changed concurrently; retry")

    updated = attendance.find_one({"_id": record["_id"]})
    return serialize_attendance(updated)


# ---------------------------------------------------------------------------
# Shared monthly aggregation
# ---------------------------------------------------------------------------

def monthly_attendance_pipeline(query):
    return [
        {"$match": query},
        {
            "$set": {
                "calendar_day": {
                    "$dateFromString": {"dateString": "$date"}
                },
                "present_status": {"$in": ["$status", PRESENCE]},
                "has_work_hours": {
                    "$and": [
                        {"$in": ["$status", PRESENCE]},
                        {"$ne": [{"$ifNull": ["$work_hours", None]}, None]},
                    ]
                },
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    "$present_status",
                                    {"$gte": [{"$dayOfWeek": "$calendar_day"}, 2]},
                                    {"$lte": [{"$dayOfWeek": "$calendar_day"}, 6]},
                                ]
                            },
                            {"$cond": ["$half_day", 0.5, 1]},
                            0,
                        ]
                    }
                },
                "leave_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
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
                "total_overtime_minutes": {
                    "$sum": {"$ifNull": ["$overtime_minutes", 0]}
                },
                "on_duty_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}
                },
                "work_hours_sum": {
                    "$sum": {
                        "$cond": ["$has_work_hours", "$work_hours", 0]
                    }
                },
                "work_hours_count": {
                    "$sum": {"$cond": ["$has_work_hours", 1, 0]}
                },
            }
        },
    ]


# ---------------------------------------------------------------------------
# Employee monthly analytics
# ---------------------------------------------------------------------------

def employee_monthly_pipeline(first, last, emp_code):
    return monthly_attendance_pipeline(month_match(first, last, emp_code))


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str,
    month: str = Query(...),
):
    validate_emp_code(emp_code)
    employee = get_employee(emp_code)
    first, last = parse_month(month)

    joined = parse_date(employee["joined_on"], "joined_on")
    working_days = (
        working_days_between(max(first, joined), last)
        if joined <= last else 0
    )

    rows = list(attendance.aggregate(employee_monthly_pipeline(first, last, emp_code)))
    row = rows[0] if rows else {}

    present_days = row.get("present_days", 0)
    attendance_pct = (
        round_half_up(present_days / working_days * 100, 4)
        if working_days else None
    )

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": round_half_up(present_days, 2),
        "leave_days": row.get("leave_count", 0),
        "late_count": row.get("late_count", 0),
        "total_late_minutes": row.get("total_late_minutes", 0),
        "total_overtime_minutes": row.get("total_overtime_minutes", 0),
        "attendance_pct": attendance_pct,
    }


# ---------------------------------------------------------------------------
# Department monthly summary
# ---------------------------------------------------------------------------

def department_summary_pipeline(first, last, department=None):
    employee_match = {"joined_on": {"$lte": last.isoformat()}}
    if department is not None:
        employee_match["department"] = department

    start_string = first.isoformat()
    end_string = last.isoformat()

    return [
        {"$match": employee_match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"code": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "date": {"$gte": start_string, "$lte": end_string},
                            "$expr": {"$eq": ["$emp_code", "$$code"]},
                        }
                    },
                    {
                        "$set": {
                            "calendar_day": {
                                "$dateFromString": {"dateString": "$date"}
                            },
                            "present_status": {"$in": ["$status", PRESENCE]},
                            "has_work_hours": {
                                "$and": [
                                    {"$in": ["$status", PRESENCE]},
                                    {"$ne": [{"$ifNull": ["$work_hours", None]}, None]},
                                ]
                            },
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {
            "$group": {
                "_id": "$department",
                "headcount_codes": {"$addToSet": "$emp_code"},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$eq": ["$logs.present_status", True]},
                                    {"$gte": [{"$dayOfWeek": "$logs.calendar_day"}, 2]},
                                    {"$lte": [{"$dayOfWeek": "$logs.calendar_day"}, 6]},
                                ]
                            },
                            {"$cond": [{"$eq": ["$logs.half_day", True]}, 0.5, 1]},
                            0,
                        ]
                    }
                },
                "work_hours_sum": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$logs.has_work_hours", True]},
                            "$logs.work_hours",
                            0,
                        ]
                    }
                },
                "work_hours_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$logs.has_work_hours", True]}, 1, 0]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$logs.late_minutes", 0]}, 0]},
                            1, 0,
                        ]
                    }
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$logs.late_minutes", 0]}
                },
                "leave_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]
                    }
                },
                "on_duty_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]
                    }
                },
            }
        },
        {
            "$set": {
                "headcount": {"$size": "$headcount_codes"},
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": ["$work_hours_count", 0]},
                        {
                            "$divide": [
                                {
                                    "$round": [
                                        {
                                            "$multiply": [
                                                {
                                                    "$divide": [
                                                        "$work_hours_sum",
                                                        "$work_hours_count",
                                                    ]
                                                },
                                                100,
                                            ]
                                        },
                                        0,
                                    ]
                                },
                                100,
                            ]
                        },
                        None,
                    ]
                },
            }
        },
        {"$match": {"headcount": {"$gt": 0}}},
        {"$sort": {"_id": 1}},
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": 1,
                "present_days": {"$round": ["$present_days", 2]},
                "avg_work_hours": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
    ]


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(...),
    department: Optional[str] = None,
):
    first, last = parse_month(month)
    pipeline = department_summary_pipeline(first, last, department)
    items = list(employees.aggregate(pipeline))

    # Apply half-up rounding to returned work-hour means.
    for item in items:
        if item["avg_work_hours"] is not None:
            item["avg_work_hours"] = round_half_up(item["avg_work_hours"], 2)
        item["present_days"] = round_half_up(item["present_days"], 2)

    return {"month": month, "items": items}


# ---------------------------------------------------------------------------
# Late leaderboard
# ---------------------------------------------------------------------------

def leaderboard_pipeline(first, last, limit, department=None):
    employee_lookup_pipeline = [
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "employee",
            }
        },
        {"$unwind": "$employee"},
    ]

    if department is not None:
        employee_lookup_pipeline.append(
            {"$match": {"employee.department": department}}
        )

    return [
        {"$match": {"date": {"$gte": first.isoformat(), "$lte": last.isoformat()}}},
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$late_minutes", 0]}
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
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        *employee_lookup_pipeline,
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {"rank": {"$rank": {}}},
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$employee.name",
                "department": "$employee.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
    ]


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    first, last = parse_month(month)
    pipeline = leaderboard_pipeline(first, last, limit, department)
    items = list(attendance.aggregate(pipeline))
    return {"month": month, "items": items}


# ---------------------------------------------------------------------------
# Department trend
# ---------------------------------------------------------------------------

def build_department_trend_pipeline(department, start, end):
    start_dt = datetime.combine(start, time.min)
    end_dt = datetime.combine(end + timedelta(days=1), time.min)

    return [
        {"$match": {"department": department}},
        {
            "$set": {
                "joined_date": {
                    "$dateFromString": {
                        "dateString": "$joined_on",
                        "format": "%Y-%m-%d",
                    }
                },
                "day": {"$literal": start_dt},
            }
        },
        {
            "$densify": {
                "field": "day",
                "partitionByFields": ["emp_code"],
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [start_dt, end_dt],
                },
            }
        },
        {
            "$set": {
                "is_joined": {"$lte": ["$joined_date", "$day"]},
                "is_working_day": {
                    "$and": [
                        {"$gte": [{"$dayOfWeek": "$day"}, 2]},
                        {"$lte": [{"$dayOfWeek": "$day"}, 6]},
                    ]
                },
                "day_string": {
                    "$dateToString": {
                        "date": "$day",
                        "format": "%Y-%m-%d",
                        "timezone": "UTC",
                    }
                },
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {
                    "code": "$emp_code",
                    "date_string": "$day_string",
                    "joined": "$is_joined",
                },
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$code"]},
                                    {"$eq": ["$date", "$$date_string"]},
                                    {"$eq": ["$$joined", True]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "status": 1,
                            "half_day": 1,
                            "late_minutes": 1,
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {"$set": {"log": {"$arrayElemAt": ["$logs", 0]}}},
        {
            "$set": {
                "headcount_contribution": {
                    "$cond": ["$is_joined", 1, 0]
                },
                "present_contribution": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_joined",
                                {"$in": ["$log.status", PRESENCE]},
                            ]
                        },
                        {"$cond": [{"$eq": ["$log.half_day", True]}, 0.5, 1]},
                        0,
                    ]
                },
                "late_contribution": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_joined",
                                {"$gt": [{"$ifNull": ["$log.late_minutes", 0]}, 0]},
                            ]
                        },
                        1,
                        0,
                    ]
                },
            }
        },
        {
            "$group": {
                "_id": "$day",
                "headcount": {"$sum": "$headcount_contribution"},
                "present_count": {"$sum": "$present_contribution"},
                "late_count": {"$sum": "$late_contribution"},
                "is_working_day": {"$first": "$is_working_day"},
            }
        },
        {
            "$set": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_working_day",
                                {"$gt": ["$headcount", 0]},
                            ]
                        },
                        {
                            "$divide": [
                                {
                                    "$floor": {
                                        "$add": [
                                            {
                                                "$multiply": [
                                                    {
                                                        "$divide": [
                                                            "$present_count",
                                                            "$headcount",
                                                        ]
                                                    },
                                                    1000000,
                                                ]
                                            },
                                            0.5,
                                        ]
                                    }
                                },
                                10000,
                            ]
                        },
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"_id": 1},
                "output": {
                    "moving_avg_raw": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, 0]},
                    }
                },
            }
        },
        {
            "$set": {
                "moving_avg_7d": {
                    "$cond": [
                        {"$ne": ["$moving_avg_raw", None]},
                        {
                            "$divide": [
                                {
                                    "$floor": {
                                        "$add": [
                                            {
                                                "$multiply": [
                                                    "$moving_avg_raw",
                                                    10000,
                                                ]
                                            },
                                            0.5,
                                        ]
                                    }
                                },
                                10000,
                            ]
                        },
                        None,
                    ]
                }
            }
        },
        {"$sort": {"_id": 1}},
        {
            "$project": {
                "_id": 0,
                "date": {
                    "$dateToString": {
                        "date": "$_id",
                        "format": "%Y-%m-%d",
                        "timezone": "UTC",
                    }
                },
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": 1,
            }
        },
    ]


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_date: date = Query(..., alias="from"),
    to_date: date = Query(..., alias="to"),
):
    if from_date > to_date:
        fail(422, "'from' must be on or before 'to'")
    if (to_date - from_date).days + 1 > 92:
        fail(422, "Trend date range cannot exceed 92 days")

    if not employees.find_one({"department": department}, {"_id": 1}):
        fail(404, "Department not found")

    pipeline = build_department_trend_pipeline(department, from_date, to_date)
    items = list(employees.aggregate(pipeline))
    return {"department": department, "items": items}


# ---------------------------------------------------------------------------
# Explain endpoint
# ---------------------------------------------------------------------------

EXPLAIN_ENDPOINTS = {
    "attendance_list",
    "employee_monthly",
    "department_summary",
    "late_leaderboard",
    "department_trend",
}


def run_explain(collection_name, pipeline):
    return db.command(
        "explain",
        {
            "aggregate": collection_name,
            "pipeline": pipeline,
            "cursor": {},
        },
        verbosity="executionStats",
    )


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
    limit: int = Query(10, ge=1, le=50),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
):
    if endpoint not in EXPLAIN_ENDPOINTS:
        fail(422, "Unsupported explain endpoint")

    skip = pagination(page, page_size)

    if endpoint == "attendance_list":
        query = attendance_list_query(emp_code, date_from, date_to, status)
        pipeline = [
            {"$match": query},
            {"$sort": {"date": -1, "emp_code": 1}},
            {"$skip": skip},
            {"$limit": page_size},
        ]
        collection_name = "attendance_logs"

    elif endpoint == "employee_monthly":
        if not emp_code or not month:
            fail(422, "employee_monthly requires emp_code and month")
        validate_emp_code(emp_code)
        get_employee(emp_code)
        first, last = parse_month(month)
        pipeline = employee_monthly_pipeline(first, last, emp_code)
        collection_name = "attendance_logs"

    elif endpoint == "department_summary":
        if not month:
            fail(422, "department_summary requires month")
        first, last = parse_month(month)
        pipeline = department_summary_pipeline(first, last, department)
        collection_name = "employees"

    elif endpoint == "late_leaderboard":
        if not month:
            fail(422, "late_leaderboard requires month")
        first, last = parse_month(month)
        pipeline = leaderboard_pipeline(first, last, limit, department)
        collection_name = "attendance_logs"

    else:
        if not department or not from_date or not to_date:
            fail(422, "department_trend requires department, from, and to")

        start = parse_date(from_date, "from")
        end = parse_date(to_date, "to")

        if end < start or (end - start).days + 1 > 92:
            fail(422, "Invalid trend date range")

        if not employees.find_one({"department": department}, {"_id": 1}):
            fail(404, "Department not found")

        pipeline = build_department_trend_pipeline(department, start, end)
        collection_name = "employees"

    try:
        explanation = run_explain(collection_name, pipeline)
    except PyMongoError as exc:
        fail(503, f"Could not explain query: {exc.__class__.__name__}")

    return {
        "endpoint": endpoint,
        "collection": collection_name,
        "explain": json_safe(explanation),
    }