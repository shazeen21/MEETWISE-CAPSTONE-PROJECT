"""Minutes of Meeting (MOM) PDF Document Generator for MeetWise AI Enterprise Platform.

Generates branded executive PDF reports containing meeting metadata, participants,
executive summaries, discussion topics, decisions table, prioritized action items,
risks, and next steps.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MOMPDFGenerator:
    """Generates corporate-grade Minutes of Meeting (MOM) PDF documents."""

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, meeting_data: Dict[str, Any], output_filename: Optional[str] = None) -> str:
        """Compile meeting intelligence into a styled PDF document.

        Args:
            meeting_data: Dictionary with meeting, speakers, intelligence, action_items, decisions.
            output_filename: Optional custom filename.

        Returns:
            Absolute path string to generated PDF file.
        """
        meeting_id = meeting_data.get("id", "meeting")
        filename = output_filename or f"MOM_{meeting_id}.pdf"
        target_path = self.output_dir / filename

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
                HRFlowable,
                KeepTogether,
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch

            doc = SimpleDocTemplate(
                str(target_path),
                pagesize=letter,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=40,
            )

            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                "MOMTitle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#1E293B"),
            )
            subtitle_style = ParagraphStyle(
                "MOMSubtitle",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=10,
                leading=14,
                textColor=colors.HexColor("#64748B"),
            )
            heading1 = ParagraphStyle(
                "MOMH1",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=13,
                leading=16,
                textColor=colors.HexColor("#0F172A"),
                spaceBefore=12,
                spaceAfter=6,
            )
            body_style = ParagraphStyle(
                "MOMBody",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=9.5,
                leading=14,
                textColor=colors.HexColor("#334155"),
            )
            table_cell = ParagraphStyle(
                "TableCell",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=8.5,
                leading=11,
                textColor=colors.HexColor("#1E293B"),
            )
            table_cell_bold = ParagraphStyle(
                "TableCellBold",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8.5,
                leading=11,
                textColor=colors.HexColor("#0F172A"),
            )
            badge_high = ParagraphStyle(
                "BadgeHigh",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8,
                textColor=colors.HexColor("#DC2626"),
            )
            badge_med = ParagraphStyle(
                "BadgeMed",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8,
                textColor=colors.HexColor("#D97706"),
            )
            badge_low = ParagraphStyle(
                "BadgeLow",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8,
                textColor=colors.HexColor("#16A34A"),
            )

            elements = []

            # 1. Header Banner
            elements.append(Paragraph("MEETWISE AI — ENTERPRISE MINUTES OF MEETING", subtitle_style))
            elements.append(Spacer(1, 4))
            title_text = meeting_data.get("title") or "Meeting Intelligence Report"
            elements.append(Paragraph(title_text, title_style))
            elements.append(Spacer(1, 8))
            elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3B82F6"), spaceAfter=10))

            # 2. Metadata Grid
            date_str = meeting_data.get("meeting_date") or str(datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
            duration_min = round(float(meeting_data.get("duration_seconds", 0.0)) / 60.0, 1)
            media_type = meeting_data.get("media_type", "Audio").upper()

            meta_data = [
                [
                    Paragraph("<b>Meeting ID:</b> " + str(meeting_data.get("id", "N/A")), body_style),
                    Paragraph("<b>Date:</b> " + str(date_str)[:19], body_style),
                ],
                [
                    Paragraph("<b>Duration:</b> " + f"{duration_min} minutes", body_style),
                    Paragraph("<b>Media Type:</b> " + media_type, body_style),
                ],
                [
                    Paragraph("<b>Recording File:</b> " + str(meeting_data.get("audio_file_name", "N/A")), body_style),
                    Paragraph("<b>Platform:</b> MeetWise AI v2.0 Enterprise", body_style),
                ],
            ]
            meta_table = Table(meta_data, colWidths=[260, 260])
            meta_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(meta_table)
            elements.append(Spacer(1, 12))

            # 3. Participants & Biometric Recognition Table
            elements.append(Paragraph("1. Identified Participants & Voice Recognition", heading1))
            speakers = meeting_data.get("speakers", [])
            spk_rows = [[
                Paragraph("Speaker", table_cell_bold),
                Paragraph("Identified Employee", table_cell_bold),
                Paragraph("Recognition Confidence", table_cell_bold),
                Paragraph("Speaking Time", table_cell_bold),
                Paragraph("Detected Accent", table_cell_bold),
            ]]
            if speakers:
                for spk in speakers:
                    conf = spk.get("confidence_score", spk.get("confidence", 1.0))
                    conf_pct = f"{conf * 100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"
                    spk_rows.append([
                        Paragraph(str(spk.get("speaker_label", spk.get("label", "N/A"))), table_cell),
                        Paragraph(str(spk.get("speaker_name", spk.get("name", "Unknown"))), table_cell_bold),
                        Paragraph(conf_pct, table_cell),
                        Paragraph(f"{round(float(spk.get('total_speaking_time', 0.0)), 1)}s", table_cell),
                        Paragraph(str(spk.get("detected_accent", "International English")), table_cell),
                    ])
            else:
                spk_rows.append([Paragraph("No speakers identified", table_cell), Paragraph("-", table_cell), Paragraph("-", table_cell), Paragraph("-", table_cell), Paragraph("-", table_cell)])

            spk_table = Table(spk_rows, colWidths=[80, 150, 110, 80, 100])
            spk_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(spk_table)
            elements.append(Spacer(1, 12))

            # 4. Executive Summary
            elements.append(Paragraph("2. Executive Summary", heading1))
            summary_text = meeting_data.get("summary") or "Meeting recording processed."
            elements.append(Paragraph(summary_text, body_style))
            elements.append(Spacer(1, 12))

            # 5. Key Decisions Table
            elements.append(Paragraph("3. Formal Decisions Reached", heading1))
            decisions = meeting_data.get("decisions", [])
            dec_rows = [[
                Paragraph("#", table_cell_bold),
                Paragraph("Decision Description", table_cell_bold),
                Paragraph("Time", table_cell_bold),
            ]]
            if decisions:
                for idx, dec in enumerate(decisions, 1):
                    dec_text = dec.get("decision", "") if isinstance(dec, dict) else getattr(dec, "decision", "")
                    ts = dec.get("timestamp", "-") if isinstance(dec, dict) else getattr(dec, "timestamp", "-")
                    dec_rows.append([
                        Paragraph(str(idx), table_cell),
                        Paragraph(dec_text, table_cell),
                        Paragraph(str(ts or "-"), table_cell),
                    ])
            else:
                dec_rows.append([Paragraph("-", table_cell), Paragraph("No formal decisions recorded.", table_cell), Paragraph("-", table_cell)])

            dec_table = Table(dec_rows, colWidths=[25, 435, 60])
            dec_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(dec_table)
            elements.append(Spacer(1, 12))

            # 6. Action Items Table
            elements.append(Paragraph("4. Action Items & Deliverables", heading1))
            actions = meeting_data.get("action_items", [])
            act_rows = [[
                Paragraph("Task Deliverable", table_cell_bold),
                Paragraph("Owner", table_cell_bold),
                Paragraph("Deadline", table_cell_bold),
                Paragraph("Priority", table_cell_bold),
                Paragraph("Status", table_cell_bold),
            ]]
            if actions:
                for act in actions:
                    task = act.get("task", "") if isinstance(act, dict) else getattr(act, "task", "")
                    owner = act.get("owner", "Unassigned") if isinstance(act, dict) else getattr(act, "owner", "Unassigned")
                    dl = act.get("deadline", "Not specified") if isinstance(act, dict) else getattr(act, "deadline", "Not specified")
                    prio = act.get("priority", "Medium") if isinstance(act, dict) else getattr(act, "priority", "Medium")
                    st = act.get("status", "pending") if isinstance(act, dict) else getattr(act, "status", "pending")

                    prio_style = badge_high if prio.lower() == "high" else (badge_med if prio.lower() == "medium" else badge_low)

                    act_rows.append([
                        Paragraph(task, table_cell),
                        Paragraph(owner or "Unassigned", table_cell_bold),
                        Paragraph(dl or "Not specified", table_cell),
                        Paragraph(prio, prio_style),
                        Paragraph(st, table_cell),
                    ])
            else:
                act_rows.append([Paragraph("No action items assigned.", table_cell), Paragraph("-", table_cell), Paragraph("-", table_cell), Paragraph("-", table_cell), Paragraph("-", table_cell)])

            act_table = Table(act_rows, colWidths=[210, 100, 90, 60, 60])
            act_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(act_table)
            elements.append(Spacer(1, 14))

            # 7. Next Steps & Sign-off
            elements.append(Paragraph("5. Organizational Follow-ups & Next Steps", heading1))
            pending = meeting_data.get("pending_issues", [])
            if pending:
                for p in pending:
                    elements.append(Paragraph(f"• {p}", body_style))
            else:
                elements.append(Paragraph("No blockers or unresolved issues recorded.", body_style))

            elements.append(Spacer(1, 20))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94A3B8"), spaceAfter=8))
            elements.append(Paragraph("Minutes generated automatically by MeetWise AI Enterprise Platform.", subtitle_style))

            # Build Document
            doc.build(elements)
            logger.info(f"Successfully compiled Minutes of Meeting PDF: {target_path}")
            return str(target_path.resolve())

        except Exception as e:
            logger.error(f"Failed to generate MOM PDF: {e}")
            raise RuntimeError(f"PDF generation failed: {e}")
