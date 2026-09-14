# Physical Setup & Vision Calibration

## 1. Table Setup (Critical for System Reliability)

```
        REACHY MINI
        (45° angle)
             ↓
         ┌──────┐
         │ HEAD │  ← Cameras point at board
         │(fix) │    at 45° downward angle
         └──────┘
              \
               \45°
                \
    ┌────────────────────────────┐
    │                            │
    │      GAME BOARD            │
    │  ┌──────────────────────┐  │
    │  │  Investigators       │  │
    │  │  Gates               │  │
    │  │  Event Cards         │  │
    │  │  Doom Track, etc     │  │
    │  └──────────────────────┘  │
    │                            │
    │  ┌──────────────────────┐  │
    │  │  CARD READER ZONE    │  │
    │  │  (face-up for Reachy)│  │
    │  │  Dimensions: TBD     │  │
    │  └──────────────────────┘  │
    │                            │
    └────────────────────────────┘
```

### Physical Constraints:

**Reachy Position:**
- [ ] Height: Cameras at ~45cm above table
- [ ] Distance from board edge: ~60-80cm
- [ ] Angle: 45° (not straight down, not flat)
- [ ] Stability: Robot COMPLETELY FIXED during vision tasks
  - No head movement during image capture
  - No body sway
  - Consider physical restraints/stand

**Card Reader Zone:**
- [ ] Location: Directly in front of Reachy (90° from Reachy's perspective, not angled)
- [ ] Size: ~20cm x 30cm (1-2 cards at a time)
- [ ] Lighting: Consistent (overhead light to avoid shadows)
- [ ] Positioning: Cards placed face-up, flat on table

**Measurement & Calibration:**
- [ ] Take ruler, measure exact distances
- [ ] Create calibration reference (checkerboard pattern on board)
- [ ] Test at different times of day (lighting changes affect vision)

---

## 2. Camera Behavior Management

### Critical Problem: Head Movement During Speech

```
SCENARIO A: Reachy Reading Board
────────────────────────────────
Timeline:
  t=0ms:   "Reading mode enabled"
  t=50ms:  Head position LOCKED
  t=100ms: Camera captures board
  t=150ms: Image analysis
  t=500ms: "Analysis complete"
  t=510ms: Head UNLOCKED, can move again
  
Result: Clean, stable image ✓

SCENARIO B: Reachy Speaking (Bad!)
────────────────────────────────
Timeline:
  t=0ms:   "Vou investigar em Arkham"
  t=100ms: Head nods left/right (natural speech)
  t=150ms: [Meanwhile, board image still being processed]
  t=200ms: Camera shifts → image corrupted
  
Result: Blurry/misaligned board view ✗
```

### Solution: State Machine with Locks

```python
class VisionController:
    """Manage when camera captures vs when Reachy moves head"""
    
    class State(Enum):
        READING = "reading"      # Head locked, capture image
        THINKING = "thinking"    # Head locked, analyzing
        SPEAKING = "speaking"    # Head free, can move for expression
        LISTENING = "listening"  # Head free, can move
    
    def __init__(self):
        self.state = State.LISTENING
        self.head_position_locked = False
    
    def begin_board_reading(self):
        """Before capturing image"""
        self.state = State.READING
        self.head_position_locked = True
        self.robot.lock_head()  # Physical/software lock
        time.sleep(0.5)  # Wait for vibrations to settle
    
    def capture_board_image(self):
        """ONLY call when locked"""
        assert self.head_position_locked, "Head must be locked!"
        image = self.robot.camera.read()
        return image
    
    def end_board_reading(self):
        """After analysis complete"""
        self.state = State.THINKING
        # Head stays locked during analysis
    
    def begin_speaking(self, text: str):
        """Unlock before TTS"""
        self.state = State.SPEAKING
        self.head_position_locked = False
        self.robot.unlock_head()
        self.robot.tts(text, voice="reachy")
    
    def begin_listening(self):
        """STT active"""
        self.state = State.LISTENING
        self.head_position_locked = False
```

---

## 3. Vision Pipeline with Movement Constraints

```
GAME LOOP
═════════════════════════════════════════════════════════

Phase 1: OBSERVATION (Head LOCKED)
──────────────────────────────────
  [ ] Lock head position
  [ ] Wait for camera to stabilize (100-200ms)
  [ ] Capture board image
  [ ] Capture card reader zone (if needed)
  [ ] Release head lock
  
Phase 2: ANALYSIS (Head can move, but careful)
──────────────────────────────────────────────
  [ ] Head: Can move during analysis (use spare compute)
  [ ] Process SAM segmentation
  [ ] Parse game state
  [ ] Query RAG for rules
  [ ] Generate action candidates
  
Phase 3: COMMUNICATION (Head FREE)
──────────────────────────────────
  [ ] Head: FREE TO MOVE (expressiveness)
  [ ] Generate explanation
  [ ] TTS with ElevenLabs
  [ ] Natural movement during speech
  
Phase 4: LISTENING (Head FREE)
──────────────────────────────
  [ ] Head: Can move during listening
  [ ] STT capturing user's words
  [ ] Diarization & addressee detection
  [ ] Wait for human to execute move
  
[Loop back to Phase 1]
```

---

## 4. Setup Phase: Two Options

You have two choices for initial game setup. Each has tradeoffs.

### OPTION A: Verbal Briefing (Faster, Less Complex)

```
HUMAN: "OK Reachy, vamos jogar contra Cthulhu. Investigadores são:"
       "Eu jogo com o Joe Diamond e a Wendy Adams."
       "Você joga com o Roland Banks."
       
REACHY: [Listening, no head movement needed]
        Queries Qdrant for: "Roland Banks - stats, abilities"
        Retrieves from memory: Joe Diamond & Wendy Adams profiles
        
REACHY: "Entendi. Vou com Roland (combate forte, sanidade 5)."
        "Joe e Wendy, vocês têm skills de investigação?"
        
[Continue briefing with setup location, initial resources, etc]

Vantagens:
  ✓ Rápido (1-2 minutos)
  ✓ Natural conversation
  ✓ Sem dependency de visão OCR
  ✓ Escalável (1 ou 10 investigadores)

Desvantagens:
  ✗ Reachy depende de você descrever tudo
  ✗ Sem visual confirmation
  ✗ Menos embodied
```

### OPTION B: Visual Reading (Complex, More Embodied)

```
1. POSITION PHASE
   ├─ You place investigator cards in "Card Reader Zone"
   │  One by one, facing Reachy
   │
   ├─ REACHY LOCKS HEAD
   │  Captures image of card
   │
   ├─ OCR/Vision extracts:
   │  - Character name
   │  - Stats (strength, sanity, lore, etc)
   │  - Skills
   │  - Starting equipment
   │
   ├─ Cross-references Qdrant for full profile
   │
   └─ Repeats for each investigator

2. ANALYSIS PHASE
   ├─ REACHY'S HEAD UNLOCKED
   │  Analyzes all character interactions
   │
   ├─ Identifies:
   │  - Synergies (who works well together?)
   │  - Weaknesses (what gaps exist?)
   │  - Team composition implications
   │
   └─ Prepares strategy briefing

3. BRIEFING PHASE
   └─ REACHY SPEAKS with full understanding of team

Vantagens:
  ✓ Visual confirmation (Reachy "saw" everyone)
  ✓ Emergent strategy (can infer from stats)
  ✓ Highly embodied
  ✓ Research-valuable (vision + reasoning)

Desvantagens:
  ✗ Slow (5-10 minutes for setup)
  ✗ Requires good card OCR
  ✗ Complex vision pipeline
  ✗ Fragile if card quality bad
```

### RECOMMENDATION FOR YOU:

Start with **OPTION A** (verbal briefing).
- Gets you playing faster
- Establishes game loop
- THEN add Option B as Phase 1.3 enhancement

---

## 5. Game State Includes ALL Investigators

```python
# IMPORTANT: Reachy tracks everyone, not just self

game_state = {
  "investigator_pool": {
    "reachy_controlled": [
      {
        "name": "Roland Banks",
        "location": "Arkham",
        "sanity": 5,
        "health": 3,
        ...
      }
    ],
    "player_controlled": [
      {
        "name": "Joe Diamond",
        "location": "Graveyard",
        "controller": "Alessandro",  # ← Important!
        ...
      },
      {
        "name": "Wendy Adams",
        "location": "Library",
        "controller": "Alessandro",
        ...
      }
    ]
  },
  
  "all_investigator_stats": {
    # Full RAG context for strategy
    "Roland Banks": {
      "strength": 4,
      "lore": 2,
      "craft": 1,
      "common_sense": 3,
      "will": 2,
      "unique_ability": "Combat specialist"
    },
    "Joe Diamond": {
      "strength": 3,
      "lore": 4,  # Better at understanding lore/mysteries
      ...
    },
    ...
  }
}
```

**Why this matters for strategy:**
- Reachy suggests: "You investigate (Joe's strength), I handle combat (Roland's strength)"
- Cross-player optimization: "Wendy covers investigation weakness"

---

## 6. Calibration Procedure

Before first game:

```
CALIBRATION CHECKLIST
═════════════════════

□ Physical Setup
  ├─ Reachy height: ___cm (measure from table to camera center)
  ├─ Distance from board: ___cm
  ├─ Angle: 45° (verify with level)
  └─ Stability: [lock Reachy to prevent sway]

□ Camera Alignment
  ├─ Capture calibration image (checkerboard on board)
  ├─ Verify: can see full board in single frame? YES/NO
  ├─ Verify: cards in reader zone are sharp? YES/NO
  └─ Adjust height/angle if needed

□ Lighting
  ├─ Natural light level: ___lux (phone app)
  ├─ Artificial lights on? YES/NO
  ├─ Shadows on board? YES/NO
  └─ Consistent time of day testing? YES/NO

□ Vision Software
  ├─ SAM segmentation test: can it detect investigator tokens?
  ├─ OCR test: can it read card text?
  ├─ Performance: FPS at full board resolution?
  └─ Failure modes documented?

□ Head Movement Protocol
  ├─ Head lock function working?
  ├─ Lock timing: ___ms (how long to stabilize?)
  ├─ Unlock timing: when is safe to move?
  └─ Test with actual TTS (does movement affect capture?)
```

---

## 7. Camera Specifications (For M1 Max + Reachy Mini)

Reachy Mini camera specs (check docs):
- [ ] Resolution: ___x___ pixels
- [ ] Frame rate: ___ FPS
- [ ] Field of view: ___ degrees
- [ ] Focal length: ___ mm

This determines:
- Max board size you can capture
- Minimum resolution for OCR
- Real-time processing feasibility

---

## 8. Test Scenarios

Before Phase 1 coding, physically test:

```
SCENARIO 1: Single Frame Capture
─────────────────────────────────
1. Place full board in view
2. Lock head
3. Capture image
4. Verify: can you see entire board?
   YES → Good setup
   NO → Increase height or distance

SCENARIO 2: Card Reading
─────────────────────────
1. Place single card in reader zone
2. Lock head
3. Capture image
4. Verify: card is sharp, readable?
   YES → Good lighting/position
   NO → Adjust card angle or lighting

SCENARIO 3: Stability Test
──────────────────────────
1. Lock head
2. Take 10 consecutive captures (no head movement)
3. Load all 10 images
4. Compare: are they identical?
   YES → Perfect
   MOSTLY → Acceptable
   NO → Find vibration source (AC, traffic, etc)

SCENARIO 4: Movement Test
─────────────────────────
1. Lock head, capture image A
2. Unlock head, move head around
3. Lock head again, capture image B
4. Compare A and B
5. Verify they frame the board identically
   YES → Head lock is reliable
   NO → Adjust mechanical lock
```

---

## 9. Reachy Head Movement Management

```python
class HeadMovementController:
    """Manage expressiveness vs vision stability"""
    
    def lock_head(self, duration_ms: int = None):
        """
        Physically lock head for vision capture
        
        Args:
            duration_ms: How long to keep locked (optional)
                        If None, manual unlock required
        """
        self.robot.head.lock()
        time.sleep(0.2)  # Mechanical settling time
        
        if duration_ms:
            time.sleep(duration_ms / 1000)
            self.unlock_head()
    
    def capture_with_stability(self, num_frames: int = 3):
        """
        Capture multiple frames and select most stable
        
        Some movement may happen even when locked
        This mitigates by choosing best frame
        """
        self.lock_head()
        frames = []
        
        for i in range(num_frames):
            frame = self.robot.camera.read()
            frames.append(frame)
            time.sleep(50)  # 50ms between captures
        
        self.unlock_head()
        
        # Return frame with best focus (sharpness metric)
        return select_sharpest_frame(frames)
    
    def expressive_speech(self, text: str):
        """
        Speak with head movement for expressiveness
        
        During this, DO NOT process vision
        (head movement will corrupt image analysis)
        """
        self.unlock_head()
        self.robot.head.enabled = True  # Allow free movement
        
        self.robot.tts(text)  # May take 1-5 seconds
        
        self.robot.head.enabled = False  # Back to control
        time.sleep(0.5)  # Settle after movement
```

---

## 10. Integration Timeline

```
WEEK 1: Physical Setup & Calibration
├─ Build table layout (45° angle)
├─ Measure exact distances
├─ Test card reader zone
├─ Document calibration values
└─ Video record setup for reference

WEEK 1-2: Phase 1 Vision
├─ Implement capture_with_stability()
├─ Test SAM on your specific board
├─ Test OCR on your cards
├─ Benchmark FPS on M1 Max
└─ Document failure modes

WEEK 2-3: Phase 2 Speech (parallel)
├─ STT pipeline
├─ Turn-taking logic
└─ Head movement during listening

[Continue to Phase 3 when both 1 & 2 work together]
```

---

## 11. Physical Space Diagram (Detailed)

```
OVERHEAD VIEW:
═════════════════════════════════════════════════════════

              REACHY MINI
                |
                | (camera wire)
                |
            ┌───┴────┐
            │  NECK  │ ← Fixed at 45°
            └───┬────┘
                │
    ┌───────────┼───────────┐
    │           │           │
    │      45° VIEW          │
    │           │           │
    │    ┌──────┴──────┐    │
    │    │   BOARD     │    │
    │    │             │    │
    │    │  [Inv]      │    │
    │    │             │    │
    │    └──────┬──────┘    │
    │           │           │
    │    ┌──────┴──────┐    │
    │    │ CARD READER │    │
    │    │    ZONE     │    │
    │    └─────────────┘    │
    └─────────────────────────┘

SIDE VIEW:
═════════════════════════════════════════════════════════

    REACHY HEAD
        │
        │ (camera lens)
        │
    ────┼──── 45°
        │
        │
        │
    ────┴──────────────
         TABLE
    
    BOARD VIEW ANGLE: ↓45° from horizontal
    MINIMUM HEIGHT: ~45cm (to see entire board)
    OPTIMAL DISTANCE: ~60-80cm from board edge
```

---

## 12. Troubleshooting Vision Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Board partially out of frame | Height too low or distance too close | Increase height/distance, test with calibration |
| Card unreadable (blurry) | Movement during capture | Use `capture_with_stability()` with multiple frames |
| Shadows on board | Light from side | Overhead lighting only, close windows |
| Inconsistent images | Vibration/sway | Check mechanical lock, test on different table |
| SAM segmentation fails | Angle too steep or too flat | Verify 45° angle, test on reference images |
| FPS too slow | Too high resolution | Resize image in processing, use GPU acceleration |

---

## Next Steps

Before coding Phase 1:

1. [ ] Build physical setup (today)
2. [ ] Measure exact distances & angles
3. [ ] Test camera field of view
4. [ ] Create calibration checklist reference
5. [ ] Document your specific setup (for reproducibility)

Then we code Phase 1.1 with these exact physical constraints in mind.

Ready to build the table?
