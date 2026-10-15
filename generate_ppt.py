import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette
    BG_DARK = RGBColor(15, 23, 42)      # #0F172A Deep Slate
    CARD_BG = RGBColor(30, 41, 59)      # #1E293B Dark Blue-Gray
    TEXT_LIGHT = RGBColor(241, 245, 249)# #F1F5F9 Off-White
    TEXT_MUTED = RGBColor(148, 163, 184)# #94A3B8 Gray
    ACCENT_BLUE = RGBColor(56, 189, 248)# #38BDF8 Sky Blue
    ACCENT_CYAN = RGBColor(45, 212, 191)# #2DD4BF Cyan
    ACCENT_AMBER = RGBColor(251, 191, 36)# #FBBF24 Yellow/Amber

    def set_slide_background(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = BG_DARK

    def add_header(slide, title_text, category_text="DATA PREPROCESSING & SIGNALS OVERVIEW"):
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.733), Inches(1.0))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = ACCENT_CYAN
        
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(24)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------
    slide_layout = prs.slide_layouts[6]
    slide1 = prs.slides.add_slide(slide_layout)
    set_slide_background(slide1)

    # Main Title Container
    title_box = slide1.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.333), Inches(3.5))
    tf1 = title_box.text_frame
    tf1.word_wrap = True

    p_sub = tf1.paragraphs[0]
    p_sub.text = "DATA ENGINEERING & SIGNALS PREPROCESSING REPORT"
    p_sub.font.size = Pt(14)
    p_sub.font.bold = True
    p_sub.font.color.rgb = ACCENT_CYAN
    p_sub.space_after = Pt(10)

    p_main = tf1.add_paragraph()
    p_main.text = "Multimodal Voice & Text Intelligence System"
    p_main.font.size = Pt(40)
    p_main.font.bold = True
    p_main.font.color.rgb = TEXT_LIGHT
    p_main.space_after = Pt(15)

    p_desc = tf1.add_paragraph()
    p_desc.text = "Detailed Breakdown of Preprocessing Workflows & Fundamental Signals and Systems Concepts Applied"
    p_desc.font.size = Pt(18)
    p_desc.font.color.rgb = TEXT_MUTED

    # Footer/Author Info
    footer_box = slide1.shapes.add_textbox(Inches(1.0), Inches(6.0), Inches(11.333), Inches(0.8))
    tf_f = footer_box.text_frame
    p_f = tf_f.paragraphs[0]
    p_f.text = "Prepared by: Member 1 (Data Engineer & Preprocessing Specialist)"
    p_f.font.size = Pt(14)
    p_f.font.color.rgb = ACCENT_BLUE

    # -------------------------------------------------------------
    # SLIDE 2: OVERVIEW OF PREPROCESSING WORKFLOW
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(slide_layout)
    set_slide_background(slide2)
    add_header(slide2, "Summary of Member 1 Tasks & Core Deliverables", "WORKFLOW OVERVIEW")

    tasks = [
        ("Task 1: Dataset Management", "Downloaded & structured Mozilla Common Voice TSV metadata and raw multilingual text corpora."),
        ("Task 2.1: Data Balancing", "Cleaned missing entries & balanced male/female classes evenly to prevent demographic model bias."),
        ("Task 2.2: Signal Standardization", "Resampled raw audio to 16 kHz and standardized durations to exactly 5 seconds (80,000 samples)."),
        ("Task 3: Text Cleaning & Encoding", "Processed raw text using Regex to strip noise and mapped language labels to numeric targets."),
        ("Task 4: Stratified Splitting", "Generated 70/15/15 Train-Val-Test splits preserving exact class proportions for team evaluation.")
    ]

    for i, (t_title, t_desc) in enumerate(tasks):
        top_pos = Inches(1.8 + i * 1.0)
        shape = slide2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), top_pos, Inches(11.733), Inches(0.85))
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = ACCENT_BLUE if i == 2 else CARD_BG

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        tf.margin_top = Inches(0.12)

        p1 = tf.paragraphs[0]
        p1.text = t_title
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = ACCENT_BLUE if i == 2 else ACCENT_CYAN

        p2 = tf.add_paragraph()
        p2.text = t_desc
        p2.font.size = Pt(12)
        p2.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 3: TASK 2.2 - AUDIO SAMPLING & DISCRETIZATION
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(slide_layout)
    set_slide_background(slide3)
    add_header(slide3, "Task 2.2: Audio Resampling & Standardized Framing", "AUDIO PREPROCESSING")

    # Left Box - What Was Done
    left_shape = slide3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    left_shape.fill.solid()
    left_shape.fill.fore_color.rgb = CARD_BG
    left_shape.line.color.rgb = CARD_BG
    tf_l = left_shape.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = Inches(0.3)

    p = tf_l.paragraphs[0]
    p.text = "WHAT WAS DONE (IMPLEMENTATION)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(10)

    bullets_l = [
        "Loaded heterogeneous audio formats (.mp3 / .wav) from the Mozilla Common Voice repository.",
        "Resampled all continuous continuous/discrete audio streams to a fixed sampling rate of fs = 16,000 Hz.",
        "Padded shorter signals with trailing zeros or truncated longer clips to yield exactly 5.0 seconds.",
        "Constructed uniform NumPy array matrices with dimension [N_samples, 80000] for downstream neural network training."
    ]
    for b in bullets_l:
        pb = tf_l.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_LIGHT
        pb.space_after = Pt(8)

    # Right Box - Signals & Systems Concept
    right_shape = slide3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.733), Inches(5.0))
    right_shape.fill.solid()
    right_shape.fill.fore_color.rgb = CARD_BG
    right_shape.line.color.rgb = ACCENT_AMBER
    tf_r = right_shape.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = Inches(0.3)

    p = tf_r.paragraphs[0]
    p.text = "SIGNALS & SYSTEMS CONCEPT APPLIED"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    p.space_after = Pt(10)

    concepts_r = [
        ("Nyquist-Shannon Sampling Theorem", "Human speech signals span 50 Hz – 7 kHz. By sampling at fs = 16 kHz, fs > 2 * B max (14 kHz), avoiding aliasing distortion."),
        ("Continuous to Discrete Conversion (C/D)", "Converts continuous acoustic pressure waves x(t) into discrete sequences x[n] = x(nTs), where Ts = 1/16000 = 62.5 µs."),
        ("Finite Length Framing & Zero Padding", "Ensures dynamic length signals x[n] map to a fixed LTI system input dimension L = 80,000 samples.")
    ]
    for title, desc in concepts_r:
        pt = tf_r.add_paragraph()
        pt.text = title
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = ACCENT_CYAN
        
        pd = tf_r.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(11)
        pd.font.color.rgb = TEXT_LIGHT
        pd.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 4: TASK 2.1 - DATA BALANCING & IMPULSE DOMAIN
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(slide_layout)
    set_slide_background(slide4)
    add_header(slide4, "Task 2.1: Data Balancing & Demographic Mapping", "AUDIO PREPROCESSING")

    left_shape = slide4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    left_shape.fill.solid()
    left_shape.fill.fore_color.rgb = CARD_BG
    left_shape.line.color.rgb = CARD_BG
    tf_l = left_shape.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = Inches(0.3)

    p = tf_l.paragraphs[0]
    p.text = "WHAT WAS DONE (IMPLEMENTATION)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(10)

    bullets_l = [
        "Parsed large TSV metadata manifests from Common Voice datasets.",
        "Filtered out corrupted records, missing metadata, and invalid gender/age tags.",
        "Applied random subsampling to balance male and female speaker distributions equally.",
        "Aligned audio filenames with normalized target vectors for supervised classification."
    ]
    for b in bullets_l:
        pb = tf_l.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_LIGHT
        pb.space_after = Pt(8)

    right_shape = slide4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.733), Inches(5.0))
    right_shape.fill.solid()
    right_shape.fill.fore_color.rgb = CARD_BG
    right_shape.line.color.rgb = ACCENT_AMBER
    tf_r = right_shape.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = Inches(0.3)

    p = tf_r.paragraphs[0]
    p.text = "SIGNALS & SYSTEMS CONCEPT APPLIED"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    p.space_after = Pt(10)

    concepts_r = [
        ("Discrete Sequence Masking & Selection", "Represented data filtering as element-wise multiplication with a binary selection window w[k]: y[k] = x[k] · w[k]."),
        ("Signal-to-Noise & Probability Distribution", "Balancing classes prevents probability distribution skew, ensuring equal prior probability P(Male) = P(Female) = 0.5 across the signal space."),
        ("Deterministic Target Vector Mapping", "Transformed continuous categorical metadata into discrete impulse vectors δ[k - c] representing system classes.")
    ]
    for title, desc in concepts_r:
        pt = tf_r.add_paragraph()
        pt.text = title
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = ACCENT_CYAN
        
        pd = tf_r.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(11)
        pd.font.color.rgb = TEXT_LIGHT
        pd.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 5: TASK 3 - TEXT CLEANING & ENCODING
    # -------------------------------------------------------------
    slide5 = prs.slides.add_slide(slide_layout)
    set_slide_background(slide5)
    add_header(slide5, "Task 3: Text Normalization & Label Encoding", "TEXT PREPROCESSING")

    left_shape = slide5.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    left_shape.fill.solid()
    left_shape.fill.fore_color.rgb = CARD_BG
    left_shape.line.color.rgb = CARD_BG
    tf_l = left_shape.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = Inches(0.3)

    p = tf_l.paragraphs[0]
    p.text = "WHAT WAS DONE (IMPLEMENTATION)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(10)

    bullets_l = [
        "Ingested raw multilingual text strings from language detection corpora.",
        "Applied Regular Expression (Regex) filters to remove punctuation, digits, brackets, and URLs.",
        "Normalized string sequences to lowercase and stripped superfluous whitespaces.",
        "Converted categorical language strings into integer targets using Scikit-Learn LabelEncoder."
    ]
    for b in bullets_l:
        pb = tf_l.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_LIGHT
        pb.space_after = Pt(8)

    right_shape = slide5.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.733), Inches(5.0))
    right_shape.fill.solid()
    right_shape.fill.fore_color.rgb = CARD_BG
    right_shape.line.color.rgb = ACCENT_AMBER
    tf_r = right_shape.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = Inches(0.3)

    p = tf_r.paragraphs[0]
    p.text = "SIGNALS & SYSTEMS CONCEPT APPLIED"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    p.space_after = Pt(10)

    concepts_r = [
        ("High-Pass / Stop-Band Filtering (Noise Reduction)", "Regex acts as a spatial stop-band filter H(e^jω) that suppresses high-frequency noise (punctuation/numbers) while passing semantic components."),
        ("Symbolic Discrete Quantization", "Quantizes a discrete alphabet of text symbols into integer codebooks, mapping continuous textual variations into finite quantization levels."),
        ("System Linearity & Invariance", "Text cleaning is a deterministic, time-invariant transformation: cleaning the same string produces identical output regardless of input time.")
    ]
    for title, desc in concepts_r:
        pt = tf_r.add_paragraph()
        pt.text = title
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = ACCENT_CYAN
        
        pd = tf_r.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(11)
        pd.font.color.rgb = TEXT_LIGHT
        pd.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 6: TASK 4 - STRATIFIED DATA SPLITTING
    # -------------------------------------------------------------
    slide6 = prs.slides.add_slide(slide_layout)
    set_slide_background(slide6)
    add_header(slide6, "Task 4: Stratified Dataset Splitting Architecture", "PIPELINE INTEGRATION")

    left_shape = slide6.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
    left_shape.fill.solid()
    left_shape.fill.fore_color.rgb = CARD_BG
    left_shape.line.color.rgb = CARD_BG
    tf_l = left_shape.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = Inches(0.3)

    p = tf_l.paragraphs[0]
    p.text = "WHAT WAS DONE (IMPLEMENTATION)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(10)

    bullets_l = [
        "Implemented a reusable train_test_split pipeline function using Scikit-Learn.",
        "Divided datasets into 70% Training, 15% Validation, and 15% Testing partitions.",
        "Enforced class stratification across gender, age, and language target vectors.",
        "Saved binary .npz arrays and metadata CSV files for Member 2, 3, and 4 handoff."
    ]
    for b in bullets_l:
        pb = tf_l.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = TEXT_LIGHT
        pb.space_after = Pt(8)

    right_shape = slide6.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.733), Inches(5.0))
    right_shape.fill.solid()
    right_shape.fill.fore_color.rgb = CARD_BG
    right_shape.line.color.rgb = ACCENT_AMBER
    tf_r = right_shape.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = Inches(0.3)

    p = tf_r.paragraphs[0]
    p.text = "SIGNALS & SYSTEMS CONCEPT APPLIED"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER
    p.space_after = Pt(10)

    concepts_r = [
        ("Stationary Ergodic Process Assumption", "Stratification guarantees that Train, Val, and Test subsets maintain identical statistical moments (mean, variance) across classes."),
        ("Ortho-System Generalization & Leakage Prevention", "Decoupling test sets ensures evaluation across non-overlapping signal subspaces, measuring system impulse response to unseen signals."),
        ("Cross-Correlation Integrity", "Preserves the cross-correlation structure between feature vector X[n] and target vector Y[n] across all dataset splits.")
    ]
    for title, desc in concepts_r:
        pt = tf_r.add_paragraph()
        pt.text = title
        pt.font.size = Pt(13)
        pt.font.bold = True
        pt.font.color.rgb = ACCENT_CYAN
        
        pd = tf_r.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(11)
        pd.font.color.rgb = TEXT_LIGHT
        pd.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 7: SIGNALS & SYSTEMS SUMMARY MATRIX
    # -------------------------------------------------------------
    slide7 = prs.slides.add_slide(slide_layout)
    set_slide_background(slide7)
    add_header(slide7, "Comprehensive Mapping: Preprocessing vs. Signals Concepts", "SUMMARY MATRIX")

    rows, cols = 5, 3
    table_shape = slide7.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.733), Inches(5.0))
    table = table_shape.table

    table.columns[0].width = Inches(2.5)
    table.columns[1].width = Inches(4.5)
    table.columns[2].width = Inches(4.733)

    headers = ["Preprocessing Task", "Python Implementation", "Signals & Systems Concept"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ACCENT_CYAN

    matrix_data = [
        ("Audio Resampling (16 kHz)", "librosa.load(..., sr=16000)", "Nyquist-Shannon Sampling Theorem & Anti-Aliasing"),
        ("Fixed Duration (5s / 80k)", "np.pad() & truncation array slice", "Finite Windowing & Discrete C/D Signal Conversion"),
        ("Text Noise Removal", "re.sub() Regex Cleaning", "Stop-Band Spatial Noise Filtering & Quantization"),
        ("Dataset Stratified Splitting", "train_test_split(..., stratify=y)", "Stationary Signal Statistics & Ergodic Preservation")
    ]

    for row_idx, data in enumerate(matrix_data, start=1):
        for col_idx, text in enumerate(data):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BG_DARK
            p = cell.text_frame.paragraphs[0]
            p.text = text
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_LIGHT if col_idx < 2 else ACCENT_AMBER

    # Save Presentation
    output_filename = "Member1_Preprocessing_Signals_Report.pptx"
    prs.save(output_filename)
    print(f"Presentation saved as '{output_filename}'")

if __name__ == "__main__":
    create_presentation()