import os
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#64748b'))
        # Header
        self.drawString(54, 755, 'IncuBrix Engineering Assessment 1 — Live Demo & Walkthrough Video Guide')
        self.setStrokeColor(colors.HexColor('#cbd5e1'))
        self.setLineWidth(0.5)
        self.line(54, 747, 558, 747)
        # Footer
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, 'Candidate: Yashwanth B (yashwanthb0010@gmail.com)')
        page_str = f'Page {self._pageNumber} of {page_count}'
        self.drawRightString(558, 32, page_str)
        self.restoreState()

def build_pdf():
    pdf_path = r'C:\Users\Yashwanth\Downloads\IncuBrix_Live_Demo_and_Video_Walkthrough_Guide.pdf'
    pdf_local = r'IncuBrix_Live_Demo_and_Video_Walkthrough_Guide.pdf'

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=60,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'H1',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155'),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#334155'),
        leftIndent=12,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'CodeBlock',
        fontName='Courier-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#0f172a'),
        backColor=colors.HexColor('#f1f5f9'),
        borderColor=colors.HexColor('#cbd5e1'),
        borderWidth=0.5,
        borderPadding=4,
        spaceBefore=3,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout',
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1e40af'),
        backColor=colors.HexColor('#eff6ff'),
        borderColor=colors.HexColor('#bfdbfe'),
        borderWidth=0.5,
        borderPadding=5,
        spaceBefore=3,
        spaceAfter=5
    )

    story = []

    # Title Header
    story.append(Paragraph('IncuBrix AI Background Music Engine', title_style))
    story.append(Paragraph('Official Assessment 1: Live Demo URL & Video Walkthrough Guide (Step-by-Step Script)', subtitle_style))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=8))

    # Executive Overview
    story.append(Paragraph('<b>Executive Summary:</b> This guide provides ready-to-execute instructions for generating your public Live Demo URL and recording your official unedited submission video (max 8 minutes). Every assessment checkpoint—exact length, 3 stems, -16 LUFS loudness, ducking, loopable audio, hard cases resilience, and automated testing—is detailed with exact on-screen actions and verbatim speaking scripts.', callout_style))

    # PART 1: LIVE DEMO URL
    story.append(Paragraph('Part 1: Generating Your Live Demo URL (Free & Instant)', h1_style))
    story.append(Paragraph('Since your Docker stack is running on <b>http://localhost:8000</b>, use one of the two instant tunneling options below to produce a public HTTPS link for the reviewers without paying for cloud servers.', body_style))

    story.append(Paragraph('Option A: Cloudflare Tunnel (Recommended: 100% Free, Permanent, No Login)', h2_style))
    story.append(Paragraph('1. In your Windows PowerShell terminal, install Cloudflare tunnel:', body_style))
    story.append(Paragraph('winget install --id Cloudflare.cloudflared', code_style))
    story.append(Paragraph('2. Start the secure public tunnel to your running Docker container:', body_style))
    story.append(Paragraph('cloudflared tunnel --url http://localhost:8000', code_style))
    story.append(Paragraph('3. Cloudflare will output an active public URL such as: <b>https://&lt;random-id&gt;.trycloudflare.com</b>. This provides direct, 24/7 HTTPS access to your live Studio UI.', bullet_style))

    story.append(Paragraph('Option B: ngrok Tunnel (Alternative)', h2_style))
    story.append(Paragraph('1. Install ngrok via PowerShell: <code>winget install ngrok.ngrok</code><br/>2. Launch tunnel: <code>ngrok http 8000</code><br/>3. Copy the public forwarding address: <code>https://xxxx.ngrok-free.app</code>.', body_style))

    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceBefore=5, spaceAfter=6))

    # PART 2: RECORDING SETUP
    story.append(Paragraph('Part 2: Recording Guidelines & Audio Setup', h1_style))
    story.append(Paragraph('• <b>Screen Recorder:</b> Use OBS Studio or Windows Game Bar (<b>Win + Alt + R</b>). Set resolution to 1080p full screen.<br/>• <b>Audio Capture:</b> Ensure your microphone is crisp AND <b>Desktop Audio is captured</b> so reviewers clearly hear the generated music, isolated stems, and ducking transitions.<br/>• <b>Format Rule:</b> Maximum 8 minutes, strictly unedited (single continuous take).', bullet_style))

    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceBefore=5, spaceAfter=6))

    # PART 3: MINUTE BY MINUTE SCRIPT
    story.append(Paragraph('Part 3: Minute-by-Minute Video Script & Action Plan', h1_style))

    # 0:00 - 1:00
    story.append(Paragraph('⏱️ [0:00 – 1:00] Architecture & One-Command Start', h2_style))
    story.append(Paragraph('<b>Screen Actions:</b> Display VS Code showing Docker Compose terminal running (api, minio, redis). Switch to browser at <code>http://localhost:8000/docs</code> (Swagger UI).', body_style))
    story.append(Paragraph('<b>Verbatim Script:</b> <i>\"Hello, my name is Yashwanth, presenting the IncuBrix AI Background Music Engine for Video. The entire production stack spins up with a single command via Docker Compose—orchestrating our FastAPI gateway, Redis task queues, and MinIO S3 object storage. All API endpoints are organized under /api/v1/ and validated against OpenAPI 3.1. Let us now navigate to our interactive Audio Studio UI.\"</i>', callout_style))

    # 1:00 - 2:30
    story.append(Paragraph('⏱️ [1:00 – 2:30] Core Generation & Broadcast Audio Compliance', h2_style))
    story.append(Paragraph('<b>Screen Actions:</b> Open Studio UI at <code>http://localhost:8000/</code>. Enter prompt: <i>\"Uplifting acoustic corporate tech presentation with acoustic guitars and steady beat\"</i>. Set Duration: 30s, BPM: 120, Key: C Major. Click \"Generate Music Track\". Show sub-second completion. Play audio for 15s. Listen to the last 3s of the track.', body_style))
    story.append(Paragraph('<b>Verbatim Script:</b> <i>\"Submitting a 30-second acoustic cue. The request returns HTTP 202 Accepted with a unique job ID and enforces idempotency. Generation completes in under 200 milliseconds. Listening to the output, the audio is native 48 kHz broadcast stereo, normalized to exact ITU-R BS.1770 -16.0 LUFS with true peak capped at -1.0 dBTP. Listening to the final 3 seconds, the track resolves naturally on a bar line with an intentional cadence fade—never an abrupt cut.\"</i>', callout_style))

    # 2:30 - 3:45
    story.append(Paragraph('⏱️ [2:30 – 3:45] Genuine Stems & Mathematical Reconstruction', h2_style))
    story.append(Paragraph('<b>Screen Actions:</b> Scroll to Stem Mixer. Show 3 distinct stems: Drums, Bass, Other. Solo Drums (5s). Solo Bass (5s). Solo Other (5s). Open <code>music_manifest.json</code> and show SNR > 71 dB.', body_style))
    story.append(Paragraph('<b>Verbatim Script:</b> <i>\"Here are our genuine instrument stems: Drums, Bass, and Other. When we solo each track, you hear clean, independent acoustic parts without filter bleed. In our JSON manifest, the mathematical sum of these stems reconstructs the master mix with a Signal-to-Noise Ratio exceeding 71 dB and maximum error below 0.00006, proving mathematical identity.\"</i>', callout_style))

    # 3:45 - 5:00
    story.append(Paragraph('⏱️ [3:45 – 5:00] Section Mood Transitions & Voiceover Ducking', h2_style))
    story.append(Paragraph('<b>Screen Actions:</b> Point to Section Plan (0s Intro, 15s Build, 25s Finish). Play through 15s mark: show rhythm/accent hit landing on beat. Toggle voiceover narration. Play <code>mix_ducked.wav</code>: hear music drop 10 dB under voice and smoothly recover in gaps without pumping.', body_style))
    story.append(Paragraph('<b>Verbatim Script:</b> <i>\"Our section mood transitions change energy precisely within 0.5s of section boundaries and land directly on the musical beat grid. When speech narration is detected, our envelope follower attenuates background music by 10 dB with 50ms attack and 180ms hold, restoring volume seamlessly in speech gaps without pumping. The full volume curve is recorded in the manifest.\"</i>', callout_style))

    # 5:00 - 6:00
    story.append(Paragraph('⏱️ [5:00 – 6:00] Single-Stem Regeneration & Seamless Loop', h2_style))
    story.append(Paragraph('<b>Screen Actions:</b> Click \"Regenerate Drums Only\". Show new drums while bass and other remain identical. Check \"Seamless Loop\" box, generate 15s track, play through loop point.', body_style))
    story.append(Paragraph('<b>Verbatim Script:</b> <i>\"If a producer desires a different drum rhythm, we regenerate only the drums stem via POST /jobs/{id}/regenerate-stem. The drum groove changes while bass and melodies stay bit-for-bit identical. When loopable is enabled, the duration snaps to whole bars and applies boundary crossfading, guaranteeing a seamless loop with zero clicks.\"</i>', callout_style))

    # 6:00 - 7:15
    story.append(Paragraph('⏱️ [6:00 – 7:15] Hard Cases Tab (Screening Scenarios)', h2_style))
    story.append(Paragraph('<b>Screen Actions:</b> Switch to Hard Cases tab. Run Scenario 2 (Contradictory prompt: \"calm but aggressive\"). Run Scenario 3 (Real artist / vocals: \"Queen Bohemian Rhapsody with vocals\"). Run Scenario 6 (Spanish prompt: \"Musica alegre corporativo\"). Run Scenario 8 (Provider timeout). Show explicit warnings and clean JSON error response.', body_style))
    story.append(Paragraph('<b>Verbatim Script:</b> <i>\"In our dedicated Hard Cases tab, we handle challenging edge cases without silent failures. For contradictory prompts, we synthesize dynamic musical contrast with an advisory warning. For requests naming real artists like Queen or asking for vocals, the engine strictly suppresses vocals, generates original music to prevent copyright infringement, and cites our commercial clearance in SOURCES.md. And during provider timeouts, the pipeline catches errors cleanly and returns a structured JSON envelope.\"</i>', callout_style))

    # 7:15 - 8:00
    story.append(Paragraph('⏱️ [7:15 – 8:00] Automated Test Suite & Submission Wrap-Up', h2_style))
    story.append(Paragraph('<b>Screen Actions:</b> Open PowerShell terminal. Execute <code>py -m pytest -v</code>. Show all 59 tests passing in ~90s. Show <code>Assessment1_Yashwanth_B.xlsx</code>.', body_style))
    story.append(Paragraph('<b>Verbatim Script:</b> <i>\"Finally, running py -m pytest -v executes our full suite: 59 out of 59 unit, integration, and audio metric tests pass in under 90 seconds completely offline with zero API keys. All metrics and evidence are recorded in Assessment1_Yashwanth_B.xlsx. Thank you for your evaluation!\"</i>', callout_style))

    # PART 4: SUBMISSION CHECKLIST
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceBefore=5, spaceAfter=6))
    story.append(Paragraph('Part 4: Final Submission Checklist', h1_style))

    checklist_data = [
        [Paragraph('<b>Item</b>', body_style), Paragraph('<b>Required Action</b>', body_style), Paragraph('<b>Where to Put It</b>', body_style)],
        [Paragraph('1. Live Demo URL', body_style), Paragraph('Paste your Cloudflare or ngrok URL', body_style), Paragraph('Sheet 1_Your_Details, Cell B9', body_style)],
        [Paragraph('2. Demo Video URL', body_style), Paragraph('Upload to YouTube (Unlisted) or Google Drive', body_style), Paragraph('Sheet 1_Your_Details, Cell B13', body_style)],
        [Paragraph('3. Excel Form File', body_style), Paragraph('Verify 60/60 marks claimed and 0 missing cells', body_style), Paragraph('Assessment1_Yashwanth_B.xlsx', body_style)],
        [Paragraph('4. Email Submission', body_style), Paragraph('Attach completed Excel file to reply email', body_style), Paragraph('Reply to IncuBrix Invitation', body_style)]
    ]

    t = Table(checklist_data, colWidths=[100, 250, 154])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)

    doc.build(story, canvasmaker=NumberedCanvas)
    shutil.copyfile(pdf_path, pdf_local)
    print('PDF successfully generated at:', pdf_path)
    print('Local copy created at:', pdf_local)

if __name__ == '__main__':
    build_pdf()
