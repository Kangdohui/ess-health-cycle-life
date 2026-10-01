"""Create the Day 1 ESS mini-project design PDF from measured EDA outputs."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "outputs/eda"
OUT = ROOT / "output/pdf/DS-MINI-Design-울산캠퍼스_3반-강도희.pdf"
NAVY = colors.HexColor("#172B4D")
BLUE = colors.HexColor("#2F6FDB")
TEAL = colors.HexColor("#168A83")
PURPLE = colors.HexColor("#7656B2")
INK = colors.HexColor("#26364D")
MUTED = colors.HexColor("#61738A")
PALE = colors.HexColor("#EAF1FC")
PALE_TEAL = colors.HexColor("#E8F5F2")
LINE = colors.HexColor("#CBD6E5")
ZEBRA = colors.HexColor("#F4F7FB")
WHITE = colors.white
BATCH_COLORS = {"Batch 1": BLUE, "Batch 2": TEAL, "Batch 3": PURPLE}

pdfmetrics.registerFont(TTFont("ArialUnicode", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"))
FONT = "ArialUnicode"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Kicker", fontName=FONT, fontSize=8, leading=11, textColor=MUTED, spaceAfter=3))
styles.add(ParagraphStyle(name="TitleKR", fontName=FONT, fontSize=23, leading=29, textColor=NAVY, spaceAfter=5))
styles.add(ParagraphStyle(name="Subhead", fontName=FONT, fontSize=10, leading=15, textColor=MUTED, spaceAfter=10))
styles.add(ParagraphStyle(name="Section", fontName=FONT, fontSize=13, leading=18, textColor=BLUE, spaceBefore=7, spaceAfter=5))
styles.add(ParagraphStyle(name="BodyKR", fontName=FONT, fontSize=9, leading=14, textColor=INK, spaceAfter=4))
styles.add(ParagraphStyle(name="SmallKR", fontName=FONT, fontSize=7.5, leading=11, textColor=MUTED, spaceAfter=2))
styles.add(ParagraphStyle(name="TableKR", fontName=FONT, fontSize=7.5, leading=10, textColor=INK))
styles.add(ParagraphStyle(name="TableHeadKR", fontName=FONT, fontSize=7.5, leading=10, textColor=NAVY))
styles.add(ParagraphStyle(name="KpiValue", fontName=FONT, fontSize=19, leading=22, textColor=NAVY, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="KpiLabel", fontName=FONT, fontSize=7, leading=10, textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="Callout", fontName=FONT, fontSize=9, leading=14, textColor=NAVY))
styles.add(ParagraphStyle(name="Diagram", fontName=FONT, fontSize=8, leading=12, textColor=NAVY, alignment=TA_CENTER))


def p(text, style="BodyKR", markup=False):
    content = str(text) if markup else escape(str(text))
    return Paragraph(content, styles[style])


def make_table(rows, widths, header=True, font=7.5, padding=5):
    converted = []
    for ri, row in enumerate(rows):
        style = "TableHeadKR" if header and ri == 0 else "TableKR"
        converted.append([p(cell, style) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("LINEBELOW", (0, -1), (-1, -1), .5, LINE),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), PALE), ("LINEBELOW", (0, 0), (-1, 0), .7, LINE)]
    for ri in range(1 if header else 0, len(rows)):
        if ri % 2 == 0:
            commands.append(("BACKGROUND", (0, ri), (-1, ri), ZEBRA))
    table.setStyle(TableStyle(commands))
    return table


def kpi(label, value, color=BLUE):
    box = Table([[p(value, "KpiValue")], [p(label, "KpiLabel")]], colWidths=[49 * mm], rowHeights=[13 * mm, 10 * mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE if color == BLUE else (PALE_TEAL if color == TEAL else colors.HexColor("#F1ECF9"))),
        ("BOX", (0, 0), (-1, -1), .6, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return box


def chart(path, width=174 * mm, height=78 * mm):
    image = Image(str(path))
    scale = min(width / image.imageWidth, height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def callout(text, fill=PALE_TEAL):
    table = Table([[p(text, "Callout")]], colWidths=[174 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill), ("BOX", (0, 0), (-1, -1), .4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def page_header(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.setFont(FONT, 8)
    canvas.drawString(21 * mm, h - 13 * mm, "ESS HEALTH  /  DATA SCIENCE MINI PROJECT")
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT, 7.5)
    canvas.drawRightString(w - 21 * mm, h - 13 * mm, "DAY 1  ·  2026.10.01")
    canvas.setStrokeColor(BLUE)
    canvas.setLineWidth(1.4)
    canvas.line(21 * mm, h - 17 * mm, w - 21 * mm, h - 17 * mm)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(.5)
    canvas.line(21 * mm, 13 * mm, w - 21 * mm, 13 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT, 7)
    canvas.drawString(21 * mm, 8 * mm, "울산캠퍼스  ·  3반  ·  U073 강도희")
    canvas.drawRightString(w - 21 * mm, 8 * mm, f"{doc.page:02d}")
    canvas.restoreState()


def make_knee_table(cycles):
    rows = [["배치", "knee 후보 cycle", "후보 Qd (Ah)", "탐색 기준"]]
    results = []
    for batch in BATCH_COLORS:
        curve = cycles[(cycles.batch == batch) & cycles.QD.between(0, 1.3)].groupby("cycle").QD.median().dropna()
        smooth = curve.rolling(15, center=True, min_periods=8).mean().dropna()
        values = smooth.to_numpy()
        x = smooth.index.to_numpy()
        if len(values) < 60:
            continue
        slope = np.diff(values)
        accel = np.diff(slope)
        eligible = np.where((x[2:] >= 50) & (x[2:] <= min(500, x[-1])))[0]
        idx = int(eligible[np.nanargmin(accel[eligible])]) + 2 if len(eligible) else int(np.nanargmin(accel)) + 2
        result = {"batch": batch, "candidate_cycle": int(x[idx]), "median_qd_ah": float(values[idx])}
        results.append(result)
        rows.append([batch, f"{x[idx]:,}", f"{values[idx]:.3f}", "15-cycle smoothing · 최대 열화 가속"])
    pd.DataFrame(results).to_csv(EDA / "knee_candidates.csv", index=False)
    return make_table(rows, [29 * mm, 34 * mm, 35 * mm, 76 * mm])


def main():
    cells_path, cycles_path = EDA / "cell_features.csv", EDA / "cycle_summary.csv"
    corr_path = EDA / "feature_cycle_life_correlations.csv"
    needed = [cells_path, cycles_path, corr_path, EDA / "01_cycle_life_by_batch.png", EDA / "02_qd_degradation_by_batch.png", EDA / "03_delta_q_voltage_by_batch.png", EDA / "04_policy_vs_life.png", EDA / "05_early_feature_relationships.png"]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing measured EDA outputs:\n" + "\n".join(missing))
    cells = pd.read_csv(cells_path)
    cycles = pd.read_csv(cycles_path)
    corr = pd.read_csv(corr_path)
    life = cells.groupby("batch").cycle_life.agg(["count", "mean", "median", "min", "max"]).round(1)
    total = int(len(cells))
    labeled_total = int(cells.cycle_life.notna().sum())
    short = int((cells.cycle_life < 500).sum())
    long = int((cells.cycle_life > 1000).sum())
    lowest = str(life["median"].idxmin())
    highest = str(life["median"].idxmax())
    shortest_per_batch = (cells.dropna(subset=["cycle_life"]).sort_values("cycle_life")
                          .groupby("batch", sort=False).head(1))
    second_batch2 = (cells[(cells.batch == "Batch 2") & cells.cycle_life.notna()]
                     .nsmallest(2, "cycle_life").tail(1))
    short_cells = pd.concat([shortest_per_batch, second_batch2]).sort_values("cycle_life")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=21 * mm, rightMargin=21 * mm, topMargin=23 * mm, bottomMargin=19 * mm, title="ESS Health | Day 1 EDA and Model Strategy", author="U073 강도희")
    story = []

    # Page 1: batch life distributions.
    story += [p("DATA EXPLORATION 01", "Kicker"), p("배치별 수명 분포", "TitleKR"), p("Cycle life · 장수명/단수명 비율 · 배치 간 분포 차이", "Subhead")]
    kpis = Table([[kpi("전체 셀", f"{total} cells", BLUE), kpi("수명 label 유효", f"{labeled_total} / {total}", TEAL), kpi("단수명 / 장수명", f"{short} / {long}", PURPLE)]], colWidths=[54 * mm] * 3, hAlign="LEFT")
    kpis.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story += [kpis, Spacer(1, 3 * mm), chart(EDA / "01_cycle_life_by_batch.png", height=80 * mm), p("배치별 분포 요약", "Section")]
    rows = [["Batch", "label 유효/전체", "평균", "중앙값", "범위", "<500", ">1,000"]]
    for batch, row in life.iterrows():
        part = cells[cells.batch == batch]
        rows.append([batch, f"{int(row['count'])}/{len(part)}", f"{row['mean']:.1f}", f"{row['median']:.1f}", f"{row['min']:.0f}-{row['max']:.0f}", f"{int((part.cycle_life < 500).sum())}", f"{int((part.cycle_life > 1000).sum())}"])
    unlabeled = total - labeled_total
    story += [make_table(rows, [24 * mm, 28 * mm, 21 * mm, 22 * mm, 29 * mm, 18 * mm, 20 * mm]), Spacer(1, 2 * mm), callout(f"중앙 수명은 {lowest}에서 가장 낮고 {highest}에서 가장 높다. label {unlabeled}개는 수명 비교·학습에서 제외한다. Batch 2는 유효 label 39개 중 28개가 500 cycle 미만이며, 최저 두 셀의 프로토콜이 달라 단일 C-rate로 설명하기 어렵다.")]
    low_rows = [["Batch", "짧은 셀", "수명", "C1", "충전 프로토콜"]]
    for _, row in short_cells.iterrows():
        low_rows.append([row.batch, row.cell_id, f"{row.cycle_life:.0f}", f"{row.C1:g}C", row.charging_policy])
    story += [p("단수명 셀 점검 · 배치별 최저 셀 + Batch 2 차순위", "Section"),
              make_table(low_rows, [24 * mm, 28 * mm, 20 * mm, 20 * mm, 82 * mm], font=7, padding=4),
              Spacer(1, 2 * mm)]

    # Page 2: capacity fade and exploratory knee candidates.
    story += [PageBreak(), p("DATA EXPLORATION 02", "Kicker"), p("방전 용량 열화", "TitleKR"), p("Cycle별 Qd · 열화 속도 · knee 후보", "Subhead"), chart(EDA / "02_qd_degradation_by_batch.png", height=108 * mm), p("배치별 용량 변화 · cycle 100 / 300 / 500", "Section")]
    qd_rows = [["Batch", "Qd@100 (Ah)", "Qd@300 (Ah)", "Qd@500 (Ah)", "100→500 변화"]]
    for batch in BATCH_COLORS:
        curve = cycles[(cycles.batch == batch) & cycles.QD.between(0, 1.3)].groupby("cycle").QD.median()
        nearest = [curve.index[np.abs(curve.index.to_numpy() - target).argmin()] for target in (100, 300, 500)]
        vals = [float(curve.loc[cycle]) for cycle in nearest]
        change = (vals[2] / vals[0] - 1) * 100
        qd_rows.append([batch, *(f"{v:.3f}" for v in vals), f"{change:+.2f}%"])
    story += [make_table(qd_rows, [30 * mm, 34 * mm, 34 * mm, 34 * mm, 42 * mm]), p("knee 후보 탐색", "Section"), make_knee_table(cycles), Spacer(1, 2 * mm), callout("Batch 2의 cycle 100→500 중앙 Qd 감소폭이 세 배치 중 가장 크다. 1.3 Ah 초과 원자료 13건은 비정상 값으로 표시해 중앙 곡선·knee 계산에서 제외했다. knee 후보는 급격한 전환점으로 확정하지 않는다.", PALE)]

    # Page 3: early voltage-capacity signal.
    story += [PageBreak(), p("DATA EXPLORATION 03", "Kicker"), p("초기 ΔQ(V) 신호", "TitleKR"), p("ΔQ(V) = Qdlin(100 cycle) - Qdlin(10 cycle) · 전압 구간별 형태 비교", "Subhead"), chart(EDA / "03_delta_q_voltage_by_batch.png", height=100 * mm)]
    dq = cells.copy()
    dq["life_group"] = np.select([dq.cycle_life < 500, dq.cycle_life > 1000], ["Short (<500)", "Long (>1000)"], default="Middle (500-1000)")
    rows = [["Batch", "수명 그룹", "셀 수", "평균 |ΔQ|", "|ΔQ| 면적", "피크 전압"]]
    for batch in BATCH_COLORS:
        for group in ["Short (<500)", "Middle (500-1000)", "Long (>1000)"]:
            part = dq[(dq.batch == batch) & (dq.life_group == group)].dropna(subset=["dq_mean_abs", "dq_area_abs", "dq_peak_voltage"])
            if len(part):
                rows.append([batch, group, str(len(part)), f"{part.dq_mean_abs.mean():.4f}", f"{part.dq_area_abs.mean():.4f}", f"{part.dq_peak_voltage.median():.2f} V"])
    b2_short = dq[(dq.batch == "Batch 2") & (dq.life_group == "Short (<500)")].dq_mean_abs.mean()
    b2_long = dq[(dq.batch == "Batch 2") & (dq.life_group == "Long (>1000)")].dq_mean_abs.mean()
    dq_ratio = b2_short / b2_long if pd.notna(b2_long) and b2_long else np.nan
    story += [p("수명 그룹별 ΔQ 요약", "Section"), make_table(rows, [24 * mm, 37 * mm, 17 * mm, 29 * mm, 28 * mm, 27 * mm]), Spacer(1, 2 * mm), callout(f"Batch 2 단수명 셀의 평균 |ΔQ|는 장수명 셀의 {dq_ratio:.1f}배다. Batch 1·3에는 500 cycle 미만 label이 없어 배치 간 형태 차이와 표본 수를 함께 고려한다.")]

    # Page 4: charging protocol associations.
    story += [PageBreak(), p("DATA EXPLORATION 04", "Kicker"), p("충전 조건과 Cycle Life", "TitleKR"), p("Charging policy · C-rate · 프로토콜별 평균 수명", "Subhead"), chart(EDA / "04_policy_vs_life.png", height=103 * mm)]
    policy_path = EDA / "charging_policy_summary.csv"
    policy = pd.read_csv(policy_path)
    corr_rows = []
    for batch, part in cells.groupby("batch"):
        valid = part[["C1", "cycle_life"]].dropna()
        rho_life = valid.corr(method="spearman").iloc[0, 1] if len(valid) > 2 else np.nan
        slope_data = part[["C1", "early_QD_slope"]].dropna()
        rho_slope = slope_data.corr(method="spearman").iloc[0, 1] if len(slope_data) > 2 else np.nan
        corr_rows.append([batch, f"{len(valid)}", f"{rho_life:+.3f}" if np.isfinite(rho_life) else "NA",
                          f"{rho_slope:+.3f}" if np.isfinite(rho_slope) else "NA", f"{part.charging_policy.nunique()}" ])
    story += [p("고속 충전 단계 C1과 수명·초기 열화 속도의 배치별 순위 상관", "Section"),
              make_table([["Batch", "유효 셀", "C1–수명 ρ", "C1–초기 Qd 기울기 ρ", "프로토콜 수"], *corr_rows], [25 * mm, 25 * mm, 32 * mm, 57 * mm, 35 * mm]),
              Spacer(1, 2 * mm), callout("C1–수명 연관은 Batch 1에서만 뚜렷하고 Batch 2·3에서는 약하다. 초기 Qd 기울기와의 관계도 배치별로 다르므로, 충전 속도의 효과를 하나의 공통 관계로 단정하지 않는다.", PALE),
              p("프로토콜별 평균 수명 양끝 · 셀 수 함께 확인", "Section")]
    extremes = [["Batch", "평균 수명 최저 프로토콜", "평균", "n", "평균 수명 최고 프로토콜", "평균", "n"]]
    for batch, part in policy.groupby("batch", sort=False):
        valid = part.dropna(subset=["mean_life"]).sort_values("mean_life")
        lo, hi = valid.iloc[0], valid.iloc[-1]
        extremes.append([batch, lo.charging_policy, f"{lo.mean_life:.0f}", str(int(lo.n)),
                         hi.charging_policy, f"{hi.mean_life:.0f}", str(int(hi.n))])
    story += [make_table(extremes, [22 * mm, 40 * mm, 16 * mm, 10 * mm, 40 * mm, 16 * mm, 10 * mm], padding=3),
              p("프로토콜별 셀 수가 작고 배치마다 구성이 다르므로 평균 차이는 탐색적 비교로 해석한다.", "SmallKR")]

    # Page 5: feature-target associations and multicollinearity.
    story += [PageBreak(), p("DATA EXPLORATION 05", "Kicker"), p("초기 신호와 수명 관계", "TitleKR"), p("첫 100 cycle 요약 피처 · ΔQ(V) 통계량 · 수명 상관 · 피처 중복", "Subhead"), chart(EDA / "05_early_feature_relationships.png", height=96 * mm)]
    rows = [["Batch", "상위 피처", "Spearman ρ", "유효 셀"]]
    for batch in BATCH_COLORS:
        top = corr[corr.batch == batch].assign(absrho=lambda d: d.spearman_r.abs()).sort_values("absrho", ascending=False).head(4)
        for _, row in top.iterrows():
            rows.append([batch, row.feature.replace("early_", "").replace("dq_", "ΔQ·"), f"{row.spearman_r:+.3f}", str(int(row.n))])
    story += [p("배치별 수명 연관 상위 피처", "Section"), make_table(rows, [26 * mm, 91 * mm, 29 * mm, 28 * mm]), Spacer(1, 3 * mm), callout("상관은 단변량 연관성으로 해석한다. 서로 비슷한 온도·용량 피처는 상관행렬로 중복을 확인하고, 모델 안에서는 fold별 피처 선택과 규제를 적용한다.")]

    # Page 6: one modeling route and leakage-safe evaluation flow.
    story += [PageBreak(), p("MODEL DESIGN", "Kicker"), p("회귀 모델 설계 전략", "TitleKR"), p("배터리 교체 계획에 필요한 cycle-life 수치 예측", "Subhead")]
    strategy = [
        ["설계 항목", "선택 및 근거"],
        ["Target", "cycle_life: EOL(용량 80%)까지 총 cycle 수 · 연속형 회귀"],
        ["관측 구간", "첫 100 cycle만 사용 · 초기 Qd/IR/온도/충전시간 요약과 ΔQ(V) 통계량"],
        ["후보 모델", "DummyRegressor 기준선 → Ridge(주 후보: 소표본·상관 피처에 규제) → 얕은 RandomForest(비선형 비교)"],
        ["주요 지표", "MAPE · 논문 기준 9.1%와 비교 · 보조: MAE, RMSE, R²"],
        ["재현성", "random_state=42 · 환경/패키지 버전 기록 · 전처리/피처 선택은 CV pipeline 내부에서 fit"],
    ]
    story += [make_table(strategy, [32 * mm, 142 * mm]), Spacer(1, 3 * mm), p("관찰된 EDA 신호와 설계 선택", "Section")]
    evidence = [
        ["EDA 관찰", "모델 설계 반영"],
        ["ΔQ 통계량과 cycle_life의 |Spearman ρ|가 배치별 최대 0.73–0.87", "첫 100 cycle의 ΔQ 통계량을 후보 피처로 추출"],
        ["ΔQ 피처끼리 상관이 높아 중복 가능", "fold 내부 SelectKBest와 Ridge 규제로 피처 수·계수 크기 제어"],
        ["C1–수명 관계가 Batch 1 −0.483, Batch 2 +0.055, Batch 3 −0.229로 달라짐", "C-rate 단독 규칙 대신 초기 관측 피처를 함께 사용"],
        ["Batch별 cycle-life 중앙값이 858.5 / 472.0 / 1,005.5로 이동", "Batch 1 hold-out으로 검증하고 Batch 2를 분리 테스트"],
    ]
    story += [make_table(evidence, [84 * mm, 90 * mm]), Spacer(1, 3 * mm), p("분할과 평가", "Section")]
    flow = Table([
        [p("Batch 1<br/>셀 단위 분할", "Diagram", markup=True), p("Batch 1 Train<br/>GroupKFold CV<br/>모델 후보 선택", "Diagram", markup=True), p("Batch 1 Valid<br/>미사용 charging policy<br/>최종 후보 확인", "Diagram", markup=True), p("Batch 2 Test<br/>최종 1회 평가<br/>논문 MAPE와 비교", "Diagram", markup=True)],
        [p("셀과 같은 policy가 train/valid 양쪽에 섞이지 않도록 group 분리", "SmallKR"), p("각 fold 안에서 결측 대체·피처 선택·스케일링 fit", "SmallKR"), p("Valid로 과적합·배치 전 성능 차이 확인", "SmallKR"), p("추가 검증은 Batch 3로 수행 가능", "SmallKR")],
    ], colWidths=[43.5 * mm] * 4)
    flow.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), PALE), ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E8F5F2")),
        ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#F1ECF9")), ("BACKGROUND", (3, 0), (3, 0), colors.HexColor("#EAF4F0")),
        ("BOX", (0, 0), (-1, -1), .5, LINE), ("INNERGRID", (0, 0), (-1, -1), .4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [flow, Spacer(1, 5 * mm), callout("필수: Batch 1 학습 · Batch 2 테스트. Batch 3는 선택 추가 평가이며, 최종 테스트 점수로 모델을 다시 고르지 않는다.", PALE)]
    story += [Spacer(1, 3 * mm), p("회귀 과제 선택", "Section"), make_table([
        ["방식", "Target 해석", "선택 이유"],
        ["회귀 · 선택", "연속형 cycle_life = EOL까지의 cycle 수", "수명 크기를 보존해 교체 시점·사용계획에 연결"],
        ["분류 · 미선택", "cutoff로 단수명/장수명 범주화", "연속 수명 정보를 잃고 cutoff에 따라 클래스 분포가 달라짐"],
    ], [30 * mm, 63 * mm, 81 * mm])]

    doc.build(story, onFirstPage=page_header, onLaterPages=page_header)
    print(OUT)


if __name__ == "__main__":
    main()
