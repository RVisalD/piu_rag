"""Create reviewable RAG data without changing the original dataset or index."""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/piu_chunks_improved.jsonl"
OUTPUT = ROOT / "data/curated"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def main():
    originals = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    active, review, archive, changes = [], [], [], []
    seen = {}
    for original in originals:
        row = dict(original)
        title, text = row["title"].strip(), row["text"].strip()
        origin = row["chunk_id"]
        row.update(title=title, text=text, original_chunk_ids=[origin])
        row["source_url"] = row.get("source_url") or None
        row["verified_at"] = row.get("verified_at") or None
        row["academic_year"] = row.get("academic_year") or None
        row["status"] = "unverified"
        # Keep source document grouping, sanitizing punctuation in identifiers.
        row["doc_id"] = slug(str(row["doc_id"]))
        reason = None
        if title == "Current Students Library" and "secret of getting ahead" in text.lower():
            reason = "Motivational quotation without useful university-service facts."
        elif title in {"Payment And Tuition Fee", "Payment And Study Duration"}:
            reason = "Payment exemption lacks its original question and conditions."
        elif title == "Scholarship Application Procedure":
            reason = "Interview procedure does not identify the scholarship or cycle."
        elif title == "Scholarship Availability":
            reason = "50% award does not identify eligible programs or conditions."
        elif "You will need :" in text:
            reason = "Requirements list is missing from the source extract."
        elif title in {"Available Majors", "Available Bachelor Programs", "Bachelor Degree Programs"}:
            reason = "Partial program list conflicts with the more complete overview; verify scope."
        elif title == "Physics Exam":
            reason = "Exam eligibility may be incomplete relative to the scholarship overview."
        elif title == "General Reasoning Exam":
            reason = "Exam eligibility differs from the newer scholarship overview; verify cycle."
        if reason:
            row.update(status="needs_review", review_reason=reason)
            review.append(row)
            changes.append({"original_chunk_id": origin, "action": "quarantined", "reason": reason})
            continue
        if title in {"Scholarship Examination Schedule", "Rector's Scholarship"}:
            row.update(status="historical", academic_year="2024" if "2024" in text else "2025-2026")
            row["text"] = "Historical information, not a current application announcement. " + text
            archive.append(row)
            changes.append({"original_chunk_id": origin, "action": "archived", "reason": "Explicit past scholarship cycle."})
            continue
        if title == "Vision, Mission And Core Values":
            row["text"] = text.split("In order for this vision")[0].strip() + " The university's core values are Responsibility, Integrity, Discovery, and Excellence (RIDE)."
            changes.append({"original_chunk_id": origin, "action": "rewritten", "reason": "Removed unfinished ending; core-value classification is supported by the educational-philosophy record."})
        if title == "Refund Policy":
            row["text"] = "At Paragon International University, application and registration fees are non-refundable."
            changes.append({"original_chunk_id": origin, "action": "rewritten", "reason": "Removed unresolved 'Yes, you can do so' reference."})
        if title == "Academic Calendar":
            row["text"] = text.replace(" For more updated information, please see Academic Calendar", "")
        key = re.sub(r"\s+", " ", row["text"]).strip().casefold()
        if key in seen:
            seen[key]["original_chunk_ids"].append(origin)
            changes.append({"original_chunk_id": origin, "action": "deduplicated", "retained_original_chunk_id": seen[key]["original_chunk_ids"][0]})
            continue
        seen[key] = row
        # Split numbered semester curricula; preserve all introductory and trailing text.
        markers = list(re.finditer(r"\bSemester\s+([IVX]+)\s*:", row["text"]))
        if "Curriculum" in title and len(markers) >= 4:
            base = title.rsplit(" - ", 1)[0]
            prefix = row["text"][:markers[0].start()].strip()
            if prefix:
                intro = dict(row, title=base + " - Curriculum Structure", text=prefix, section="curriculum_overview")
                active.append(intro)
            for i, marker in enumerate(markers):
                end = markers[i + 1].start() if i + 1 < len(markers) else len(row["text"])
                semester = marker.group(1)
                body = row["text"][marker.start():end].strip()
                active.append(dict(row, title=f"{base} - Semester {semester} Curriculum", text=f"{base}. {body}", section="curriculum", semester=semester))
            changes.append({"original_chunk_id": origin, "action": "split_curriculum", "semester_sections": len(markers), "overview_section": bool(prefix)})
        else:
            row["section"] = title.rsplit(" - ", 1)[-1].lower() if " - " in title else "general"
            active.append(row)
    for records in (active, review, archive):
        orders = Counter()
        for i, row in enumerate(records):
            row["chunk_id"] = i
            row["section_order"] = orders[row["doc_id"]]
            orders[row["doc_id"]] += 1
    OUTPUT.mkdir(exist_ok=True)
    for name, records in [("piu_chunks_clean.jsonl", active), ("piu_chunks_review.jsonl", review), ("piu_chunks_historical.jsonl", archive)]:
        assert len({r["chunk_id"] for r in records}) == len(records)
        assert all(r["title"] and r["text"] for r in records)
        (OUTPUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    covered = {i for r in active + review + archive for i in r["original_chunk_ids"]}
    assert covered == {r["chunk_id"] for r in originals}, "Every input record must be accounted for"
    audit = {"input_records": len(originals), "clean_records": len(active), "review_records": len(review), "historical_records": len(archive), "changes": changes}
    (OUTPUT / "changes.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "changes"}, indent=2))


if __name__ == "__main__":
    main()
