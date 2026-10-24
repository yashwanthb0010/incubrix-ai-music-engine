import os
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
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
        self.setFont('Helvetica-Bold', 8)
        self.setFillColor(colors.HexColor('#475569'))
        # Running Header
        self.drawString(54, 755, 'INCUBRIX ASSESSMENT 1 — OFFICIAL VIDEO RECORDING SCRIPT & TELEPROMPTER')
        self.setStrokeColor(colors.HexColor('#cbd5e1'))
        self.setLineWidth(0.75)
        self.line(54, 747, 558, 747)
        # Running Footer
        self.line(54, 45, 558, 45)
        self.setFont('Helvetica', 8)
        self.drawString(54, 32, 'Presenter: Yashwanth B | IncuBrix AI Music Engine | Max Duration: 8:00 (Single Take)')
        page_str = f'Page {self._pageNumber} of {page_count}'
        self.drawRightString(558, 32, page_str)
        self.restoreState()

def generate_teleprompter_pdf():
    pdf_path = r'C:\Users\Yashwanth\Downloads\IncuBrix_Video_Recording_Script_and_Cue_Sheet.pdf'
    pdf_local = r'IncuBrix_Video_Recording_Script_and_Cue_Sheet.pdf'

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=58,
        bottomMargin=52
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=8
    )

    time_badge_style = ParagraphStyle(
        'TimeBadge',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#ffffff')
    )

    header_cell_style = ParagraphStyle(
        'HeaderCell',
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )

    action_style = ParagraphStyle(
        'ActionStyle',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )

    speech_style = ParagraphStyle(
        'SpeechStyle',
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#0f172a')
    )

    check_style = ParagraphStyle(
        'CheckStyle',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#15803d')
    )

    pre_box_style = ParagraphStyle(
        'PreBox',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1e40af'),
        backColor=colors.HexColor('#eff6ff'),
        borderColor=colors.HexColor('#bfdbfe'),
        borderWidth=0.75,
        borderPadding=6,
        spaceBefore=2,
        spaceAfter=6
    )

    story = []

    # Title
    story.append(Paragraph('IncuBrix AI Background Music Engine', title_style))
    story.append(Paragraph('Official Assessment 1: Video Recording Teleprompter & Step-by-Step Cue Sheet', subtitle_style))
    story.append(HRFlowable(width='100%', thickness=1.5, color=colors.HexColor('#0284c7'), spaceBefore=0, spaceAfter=6))

    # Pre-Recording Checklist Box
    story.append(Paragraph(
        '<b>🔴 PRE-RECORDING CHECKLIST (Do this 2 minutes before pressing record):</b><br/>'
        '1. <b>Docker Running:</b> Make sure your terminal has <code>docker compose up</code> running.<br/>'
        '2. <b>Browser Tabs Ready:</b> Have Tab 1: <code>http://localhost:8000/</code> (Studio UI) | Tab 2: <code>http://localhost:8000/docs</code> (Swagger UI).<br/>'
        '3. <b>Recorder Settings:</b> Open OBS or Windows Game Bar (<b>Win + Alt + R</b>). Ensure <b>Desktop Audio is captured</b> so reviewer can hear the music!<br/>'
        '4. <b>Rule:</b> Maximum 8 minutes, strictly continuous (unedited single take). Target time: ~7:15 minutes.',
        pre_box_style
    ))
    story.append(Spacer(1, 4))

    scenes = [
        {
            'time': 'PART 1: 0:00 – 0:50 (50s)',
            'title': 'Stack Architecture & One-Command Start',
            'actions': '1. Start video on VS Code terminal showing Docker running (api, minio, redis).<br/>2. Switch to browser tab at <b>http://localhost:8000/docs</b>.<br/>3. Scroll through Swagger docs showing all routes strictly under <b>/api/v1/</b>.<br/>4. Switch to Studio UI at <b>http://localhost:8000/</b>.',
            'speech': '\"Hello everyone, my name is Yashwanth, presenting the IncuBrix AI Background Music Engine for Video. The entire production stack spins up with a single command using Docker Compose—orchestrating our FastAPI gateway, Redis asynchronous task queue, and MinIO S3-compatible storage. All API endpoints are organized under /api/v1/ with complete OpenAPI 3.1 specifications. Let us now jump into our interactive Audio Studio interface.\"',
            'verifies': '• One-command start (Requirement 1)<br/>• Documented HTTP API /api/v1/ (Requirement 2)'
        },
        {
            'time': 'PART 2: 0:50 – 2:15 (85s)',
            'title': 'Core Music Generation & Broadcast Audio Compliance',
            'actions': '1. In Prompt box, enter: <i>\"Uplifting corporate presentation with acoustic guitars and steady beat\"</i>.<br/>2. Set Duration: <b>30.0s</b> | BPM: <b>120</b> | Key: <b>C Major</b>.<br/>3. Click <b>\"Generate Music Track\"</b>.<br/>4. Point out HTTP 202 Accepted and fast generation in under 200ms.<br/>5. <b>Click PLAY</b> and let the music play for 10–12 seconds.<br/>6. Scrub to <b>last 3 seconds (27s to 30s)</b> and let it play out to demonstrate the cadence fade.',
            'speech': '\"We will submit a 30-second acoustic corporate track. Notice that the POST request returns HTTP 202 Accepted with a unique job ID and enforces safe idempotency retries. Generation completes in under 200 milliseconds. Listening to the audio, the track is delivered in broadcast 48 kHz stereo, normalized to exact ITU-R BS.1770 -16.0 LUFS with true peak capped at -1.0 dBTP. Listening closely to the final 3 seconds, the track resolves naturally on a bar line with an intentional cadence fade—never a hard cut.\"',
            'verifies': '• Exact length within 0.05s with bar-line fade (Must-Have 1)<br/>• 48 kHz, -16 LUFS, <= -1 dBTP (Must-Have 5)<br/>• Background jobs & safe retries (Req 3 & 4)'
        },
        {
            'time': 'PART 3: 2:15 – 3:30 (75s)',
            'title': 'Genuine Stems & Mathematical Reconstruction Audit',
            'actions': '1. Scroll down to the <b>Stem Mixer</b>.<br/>2. Show the 3 distinct stems: <b>Drums</b>, <b>Bass</b>, and <b>Other</b>.<br/>3. <b>Solo Drums</b> and play for 5 seconds.<br/>4. <b>Solo Bass</b> and play for 5 seconds.<br/>5. <b>Solo Other</b> (guitars/synths) and play for 5 seconds.<br/>6. Click <b>\"View Manifest\"</b> and highlight <code>stem_reconstruction.snr_db > 71 dB</code>.',
            'speech': '\"Next is genuine stem separation. Here we have at least 3 distinct, independently playable stems: Drums, Bass, and Other. When we solo each one, you hear pure isolated instruments, not arbitrary frequency filters. Furthermore, looking at our music manifest, the mathematical sum of these stems reconstructs the master mix with a Signal-to-Noise Ratio exceeding 71 dB and maximum error below 0.00006, proving mathematical identity.\"',
            'verifies': '• Real stems (drums, bass, other) (Must-Have 2)<br/>• Mix equals sum of stems (SNR > 71 dB)<br/>• Instrumental only, no vocals (Must-Have 3)'
        },
        {
            'time': 'PART 4: 3:30 – 4:45 (75s)',
            'title': 'Section Mood Transitions & Voiceover Narration Ducking',
            'actions': '1. Point to the <b>Section Plan</b> (0s Intro, 15s Energetic Build, 25s Finish).<br/>2. Play across the 15.0s timestamp to hear the rhythm and crash cymbal land on the exact beat.<br/>3. Enable <b>Voiceover Ducking</b> (or select sample voice narration).<br/>4. Play <code>mix_ducked.wav</code>: hear music drop 10 dB under voice and smoothly recover in gaps with zero pumping.<br/>5. Point to the <code>volume_curve</code> in the manifest.',
            'speech': '\"Now let us look at section transitions and voiceover ducking. Our section mood changes change energy within 0.5s of boundaries and land precisely on the beat grid. When speech narration is present, our envelope detector attenuates background music by 10 dB under speech with 50ms attack and 180ms hold, recovering smoothly in speech gaps without pumping distortion. The full volume ducking curve is published in the manifest.\"',
            'verifies': '• Section plan beat alignment (Advanced Level)<br/>• Anti-pumping voiceover ducking (8-15 dB drop)<br/>• Beat & bar grid in manifest'
        },
        {
            'time': 'PART 5: 4:45 – 5:45 (60s)',
            'title': 'Single-Stem Regeneration & Seamless Looping',
            'actions': '1. Click <b>\"Regenerate Drums Only\"</b>.<br/>2. Play the updated track: show that drums groove changed, but bass and melody stems stayed identical.<br/>3. Check <b>\"Seamless Loop\"</b>, generate a 12s cue, and play continuously across the boundary.',
            'speech': '\"If a creator wants a new drum rhythm while keeping the same melody, we can regenerate just one stem via POST /jobs/{id}/regenerate-stem. The drums are re-synthesized while bass and melody remain 100% bit-for-bit identical. When loopable is checked, the track snaps to integer bar counts and applies boundary crossfading for a seamless loop with zero clicks.\"',
            'verifies': '• Change one part / stem regeneration (Advanced)<br/>• Seamless loop with zero click (Advanced)'
        },
        {
            'time': 'PART 6: 5:45 – 6:45 (60s)',
            'title': 'Hard Cases Tab (Screening Scenarios Resilience)',
            'actions': '1. Switch to the <b>\"Hard Cases\"</b> tab in the UI.<br/>2. Run Scenario 2: <i>\"calm but aggressive\"</i> -> show dynamic contrast and warning.<br/>3. Run Scenario 3: <i>\"Queen Bohemian Rhapsody with vocals\"</i> -> show vocal suppression & copyright warning.<br/>4. Run Scenario 6: Non-English prompt -> show translated genre.<br/>5. Run Scenario 8: Provider timeout -> show clean JSON error envelope.',
            'speech': '\"Here is our dedicated Hard Cases tab, built to prove resilience against the 5 screening tests. For contradictory prompts like calm but aggressive, we synthesize dynamic musical contrast with an explicit warning. When asked for real songs or vocals like Queen, the engine suppresses vocals, generates original music to prevent copyright infringement, and links to our commercial clearance in SOURCES.md. And when models fail or time out, the pipeline catches it gracefully and returns a structured JSON error envelope.\"',
            'verifies': '• Safe to publish commercially & SOURCES.md (Must-Have 4)<br/>• Hard cases handling without silent failure'
        },
        {
            'time': 'PART 7: 6:45 – 7:30 (45s)',
            'title': 'Automated Test Suite & Final Signoff',
            'actions': '1. Switch back to PowerShell terminal.<br/>2. Run: <code>py -m pytest -v</code>.<br/>3. Show tests executing and finishing: <b>59 passed in ~90s</b>.<br/>4. Show <code>Assessment1_Yashwanth_B.xlsx</code> file in VS Code.<br/>5. Thank the reviewer and stop recording.',
            'speech': '\"Finally, running py -m pytest -v executes our full automated test suite: 59 out of 59 unit, integration, and audio metric tests pass in under 90 seconds completely offline with zero API keys. All metrics, CLAP evaluations, and the completed submission form are in Assessment1_Yashwanth_B.xlsx. Thank you for your time and evaluation!\"',
            'verifies': '• 59/59 automated tests passing (Integration Req 1)<br/>• Self-assessed 60/60 marks claimed'
        }
    ]

    for idx, sc in enumerate(scenes, start=1):
        # Section Header Banner
        header_table = Table(
            [[Paragraph(f"<b>{sc['time']} — {sc['title']}</b>", time_badge_style)]],
            colWidths=[512]
        )
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1e3a8a')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        
        # Details Table
        content_data = [
            [
                Paragraph('<b>🖥️ WHAT TO SHOW & DO (ON SCREEN):</b>', header_cell_style),
                Paragraph(sc['actions'], action_style)
            ],
            [
                Paragraph('<b>🎙️ VERBATIM SCRIPT (WHAT TO SAY):</b>', header_cell_style),
                Paragraph(sc['speech'], speech_style)
            ],
            [
                Paragraph('<b>✅ WHAT THIS VERIFIES:</b>', header_cell_style),
                Paragraph(sc['verifies'], check_style)
            ]
        ]
        
        content_table = Table(content_data, colWidths=[130, 382])
        content_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
            ('BACKGROUND', (1,1), (1,1), colors.HexColor('#fefce8')), # Highlight speech in warm yellow
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        
        story.append(KeepTogether([header_table, content_table]))
        story.append(Spacer(1, 7))

    doc.build(story, canvasmaker=NumberedCanvas)
    shutil.copyfile(pdf_path, pdf_local)
    print('Cue Sheet PDF generated at:', pdf_path)
    print('Local copy created at:', pdf_local)

if __name__ == '__main__':
    generate_teleprompter_pdf()
