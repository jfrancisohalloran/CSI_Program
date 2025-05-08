import re
import os, sys
import logging
from datetime import datetime
import pandas as pd

if getattr(sys, "frozen", False):
    BASE_DIR = os.getcwd()
else:
    BASE_DIR = os.getenv(
        "ATT_PIPE_BASE",
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    )

DATE_COL_PATTERN = re.compile(r"^[A-Z][a-z]{2}\s+\d{2}$")
TIME_CHILD_ROOM_PATTERN = re.compile(
    r"(\d{1,2}:\d{2}\s?(?:AM|PM))\s*\((.*?)\)(?:\s*\[(.*?)\])?"
)

logger = logging.getLogger(__name__)


def extract_year_from_filename(filename: str) -> str | None:
    match = re.search(r"(20\d{2})", filename)
    if match:
        year = match.group(1)
        logger.debug("extract_year_from_filename: extracted %s from %s", year, filename)
        return year
    logger.warning("extract_year_from_filename: no year in %s", filename)
    return None


def parse_event(cell_text: str) -> list[tuple[str, str, str]]:
    if not isinstance(cell_text, str):
        logger.debug("parse_event: non-str input %r", cell_text)
        return []

    events: list[tuple[str, str, str]] = []
    for line in cell_text.split("\n"):
        for time_str, teacher, event_room in TIME_CHILD_ROOM_PATTERN.findall(line):
            ts = time_str.strip()
            tch = teacher.strip()
            rm  = event_room.strip()
            events.append((ts, tch, rm))
    logger.debug("parse_event: found %d events in cell", len(events))
    return events


def combine_date_time(date_obj: datetime, time_str: str) -> pd.Timestamp:
    if date_obj is None or not isinstance(time_str, str):
        logger.debug("combine_date_time: invalid inputs %r, %r", date_obj, time_str)
        return pd.NaT

    dt_str = f"{date_obj:%Y-%m-%d} {time_str}"
    try:
        ts = datetime.strptime(dt_str, "%Y-%m-%d %I:%M %p")
        logger.debug("combine_date_time: parsed %s to %s", dt_str, ts)
        return ts
    except ValueError:
        logger.warning("combine_date_time: failed to parse %s", dt_str)
        return pd.NaT


def get_level_from_room(assigned_room: str, place: str) -> str | None:
    if not assigned_room:
        logger.debug("get_level_from_room: empty room for place %s", place)
        return None

    if place == "ECEC":
        if re.search(r"\bCamp\b", assigned_room, re.IGNORECASE):
            logger.debug("get_level_from_room: mapping %s to Preschool", assigned_room)
            return "Preschool"
        
        m = re.search(r"(Infant|Toddlers?|Multi[-\s]?Age|Preschool|Pre[-\s]?K)",
                      assigned_room, re.IGNORECASE)
        if m:
            lvl = m.group(1).title()
            lvl = lvl.replace("Toddlers", "Toddler").replace("Multi Age", "Multi-Age").replace("Pre K", "Pre-K")
            logger.debug("get_level_from_room: ECEC %s to %s", assigned_room, lvl)
            return lvl
        logger.debug("get_level_from_room: no ECEC level match for %s", assigned_room)
        return None

    spellman_map = {
        "Goodnight Moon": "Infant",
        "House Pooh":     "Infant",
        "Panda Bear":     "Toddler",
        "Pandas":         "Toddler",
        "Rabbits":        "Toddler",
        "Monkeys":        "Toddler",
        "Caterpillars":   "Multi-Age",
        "Hungry Caterpillars": "Multi-Age",
        "Llama Llama":    "Multi-Age",
        "Llamas Llamas":  "Multi-Age",
        "Wild Things":    "Preschool",
        "Rainbow Fish":   "Preschool",
        "Dinosaurs":      "Pre-K",
        "Dinosaur Stomp": "Pre-K"
    }
    if assigned_room in spellman_map:
        lvl = spellman_map[assigned_room]
        logger.debug("get_level_from_room: Spellman %s to %s", assigned_room, lvl)
        return lvl

    m = re.search(r"(Infant|Toddlers?|Multi[-\s]?Age|Preschool|Pre[-\s]?K)",
                  assigned_room, re.IGNORECASE)
    if m:
        lvl = m.group(1).title()
        lvl = lvl.replace("Toddlers", "Toddler").replace("Multi Age", "Multi-Age").replace("Pre K", "Pre-K")
        logger.debug("get_level_from_room: fallback %s to %s", assigned_room, lvl)
        return lvl

    logger.debug("get_level_from_room: unmapped room %s", assigned_room)
    return None
