import os
import re
import math
import pandas as pd
from datetime import datetime

from .utils import (
    BASE_DIR,
    DATE_COL_PATTERN,
    extract_year_from_filename,
    parse_event,
    combine_date_time,
    get_level_from_room
)
from .logger import get_logger

logger = get_logger(__name__)

FILENAME_PATTERN = re.compile(r'^(ECEC|Spellman).*Student Sign.*\.xlsx$', re.IGNORECASE)

def parse_and_aggregate_attendance(force_refresh: bool = False) -> pd.DataFrame:
    base_dir = BASE_DIR
    output_excel = os.path.join(base_dir, "Grouped_Staff_Requirements.xlsx")

    if os.path.exists(output_excel):
        if force_refresh:
            logger.info("Force-refresh enabled; deleting cache %s", output_excel)
            os.remove(output_excel)
        else:
            logger.info("Loading cached attendance output: %s", output_excel)
            return pd.read_excel(output_excel)

    all_files = os.listdir(base_dir)
    excel_files = [
        os.path.join(base_dir, fn)
        for fn in all_files
        if FILENAME_PATTERN.match(fn)
    ]
    if not excel_files:
        logger.warning("No attendance files found in %s", base_dir)
        return pd.DataFrame()
    logger.info("Found %d attendance files", len(excel_files))

    def parse_file(filepath: str) -> pd.DataFrame:
        logger.debug("Parsing file %s", filepath)
        df = pd.read_excel(filepath, header=5, dtype=str)
        basename = os.path.basename(filepath)
        year = extract_year_from_filename(basename)
        place = ('Spellman' if 'Spellman' in basename else
                 'ECEC' if 'ECEC' in basename else None)

        df = df.rename(columns={"Record ID":"StudentID", "Room":"AssignedRoom"})
        data_cols = df.columns[7:]

        pairs = []
        i = 0
        while i < len(data_cols) - 1:
            col_in = str(data_cols[i]).strip()
            if DATE_COL_PATTERN.match(col_in):
                pairs.append((col_in, data_cols[i+1]))
                i += 2
            else:
                i += 1

        records = []
        for col_in, col_out in pairs:
            date_str = f"{col_in} {year}"
            try:
                date_obj = datetime.strptime(date_str, "%b %d %Y")
            except ValueError:
                logger.debug("Invalid date %s", date_str)
                continue

            for _, row in df.iterrows():
                ins_list  = parse_event(row.get(col_in, ""))
                outs_list = parse_event(row.get(col_out, ""))
                if not ins_list or not outs_list:
                    continue

                for (time_in, _, room_in), (time_out, _, _) in zip(ins_list, outs_list):
                    dt_in  = combine_date_time(date_obj, time_in)
                    dt_out = combine_date_time(date_obj, time_out)
                    if pd.isna(dt_in) or pd.isna(dt_out):
                        continue

                    duration = (dt_out - dt_in).total_seconds() / 3600
                    assigned = room_in if place == 'ECEC' else row.get('AssignedRoom','')
                    records.append({
                        'StudentID':     row.get('StudentID',''),
                        'AssignedRoom':  assigned,
                        'Tags':          row.get('Tags',''),
                        'Date':          date_obj,
                        'DurationHours': duration,
                        'Year':          year,
                        'Place':         place,
                        'SourceFile':    basename
                    })
        if not records:
            return pd.DataFrame()
        df_parsed = pd.DataFrame(records)
        return df_parsed.groupby(
            ['StudentID','Date','AssignedRoom','Tags','Year','Place','SourceFile'],
            as_index=False
        )['DurationHours'].sum()

    parsed_dfs = [df for df in (parse_file(f) for f in excel_files) if not df.empty]
    if not parsed_dfs:
        logger.warning("No valid attendance records parsed")
        return pd.DataFrame()
    combined = pd.concat(parsed_dfs, ignore_index=True)
    logger.info("Combined parsed rows: %d", combined.shape[0])

    daily = combined.groupby(
        ['Year','Place','Date','AssignedRoom'],
        as_index=False
    ).agg(
        TotalDurationHours=('DurationHours','sum'),
        StudentCount=('StudentID','nunique')
    )
    daily['FTE_Students'] = daily['TotalDurationHours'] / 9.0
    daily['Level'] = daily.apply(
        lambda r: get_level_from_room(r['AssignedRoom'], r['Place']), axis=1
    )
    lvl_ratio = {'Infant':4,'Multi-Age':4,'Toddler':6,'Preschool':10,'Pre-K':12}
    daily['StaffRatio'] = daily['Level'].map(lvl_ratio)
    daily['StaffRequired'] = daily.apply(
        lambda r: math.ceil(r['FTE_Students']/r['StaffRatio'])
                    if pd.notna(r['StaffRatio']) and r['StaffRatio']>0 else None,
        axis=1
    )

    daily.to_excel(output_excel, index=False)
    logger.info("Saved aggregated output to %s", output_excel)
    return daily
