import { useState, useEffect, useRef } from "react";

const SCREENS = {
  WELCOME: "welcome",
  PREP: "prep",
  POSITION: "position",
  CAPTURE: "capture",
  ANALYZING: "analyzing",
  RESULT: "result",
};

// ─── Pulse ring animation component ──────────────────────────────────────────
function PulseRing({ color = "#00E5C8", size = 120 }) {
  return (
    <div style={{ position: "relative", width: size, height: size, display: "flex", alignItems: "center", justifyContent: "center" }}>
      {[0, 1, 2].map((i) => (
        <div key={i} style={{
          position: "absolute",
          width: size,
          height: size,
          borderRadius: "50%",
          border: `2px solid ${color}`,
          opacity: 0,
          animation: `pulseRing 2.4s ease-out infinite`,
          animationDelay: `${i * 0.8}s`,
        }} />
      ))}
    </div>
  );
}

// ─── Step dot indicator ───────────────────────────────────────────────────────
function StepDots({ total, current }) {
  return (
    <div style={{ display: "flex", gap: 6, justifyContent: "center" }}>
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} style={{
          width: i === current ? 20 : 7,
          height: 7,
          borderRadius: 4,
          background: i === current ? "#00E5C8" : i < current ? "#00E5C850" : "#1E3A4A",
          transition: "all 0.3s ease",
        }} />
      ))}
    </div>
  );
}

// ─── Camera viewfinder overlay ────────────────────────────────────────────────
function Viewfinder({ status }) {
  const colors = {
    idle: "#00E5C8",
    good: "#00FF88",
    bad: "#FF4757",
    capturing: "#FFD32A",
  };
  const c = colors[status] || colors.idle;
  const size = 130;
  const corner = 18;
  const thick = 3;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      {/* Corner brackets */}
      {[
        { top: 0, left: 0, borderTop: `${thick}px solid ${c}`, borderLeft: `${thick}px solid ${c}` },
        { top: 0, right: 0, borderTop: `${thick}px solid ${c}`, borderRight: `${thick}px solid ${c}` },
        { bottom: 0, left: 0, borderBottom: `${thick}px solid ${c}`, borderLeft: `${thick}px solid ${c}` },
        { bottom: 0, right: 0, borderBottom: `${thick}px solid ${c}`, borderRight: `${thick}px solid ${c}` },
      ].map((s, i) => (
        <div key={i} style={{ position: "absolute", width: corner, height: corner, ...s }} />
      ))}
      {/* Center circle target */}
      <div style={{
        position: "absolute",
        top: "50%", left: "50%",
        transform: "translate(-50%,-50%)",
        width: 60, height: 60,
        borderRadius: "50%",
        border: `2px dashed ${c}80`,
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: c, opacity: 0.9 }} />
      </div>
      {/* Crosshair lines */}
      <div style={{ position: "absolute", top: "50%", left: corner + 4, right: corner + 4, height: 1, background: `${c}40` }} />
      <div style={{ position: "absolute", left: "50%", top: corner + 4, bottom: corner + 4, width: 1, background: `${c}40` }} />
    </div>
  );
}

// ─── SCREEN 1: Welcome ────────────────────────────────────────────────────────
function WelcomeScreen({ onNext }) {
  return (
    <div style={{ ...S.screen, background: "linear-gradient(160deg, #061B28 0%, #0A2E3F 60%, #062030 100%)", justifyContent: "space-between", padding: "22px 28px 20px" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ ...S.logoMark }}>
          <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
            <circle cx="13" cy="13" r="12" stroke="#00E5C8" strokeWidth="1.5" />
            <path d="M7 13 C7 9 10 7 13 7 C16 7 19 9 19 13 C19 17 16 20 13 20 C10 20 7 18 7 15" stroke="#00E5C8" strokeWidth="1.5" fill="none" strokeLinecap="round" />
            <circle cx="13" cy="13" r="3" fill="#00E5C8" opacity="0.8" />
          </svg>
        </div>
        <div>
          <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 18, fontWeight: 700, color: "#E8F8FF", letterSpacing: "0.05em" }}>SkinSense</div>
          <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: "#00E5C880", letterSpacing: "0.2em", textTransform: "uppercase" }}>AI Skin Imaging</div>
        </div>
        <div style={{ marginLeft: "auto", background: "#00E5C815", border: "1px solid #00E5C830", borderRadius: 6, padding: "3px 8px" }}>
          <span style={{ fontFamily: "monospace", fontSize: 9, color: "#00E5C8", letterSpacing: "0.1em" }}>READY</span>
        </div>
      </div>

      {/* Center illustration */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}>
        <div style={{ position: "relative", width: 110, height: 110, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <PulseRing color="#00E5C8" size={110} />
          <div style={{
            position: "absolute",
            width: 72, height: 72, borderRadius: "50%",
            background: "linear-gradient(135deg, #0E3A50, #092230)",
            border: "2px solid #00E5C840",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
              <rect x="5" y="8" width="26" height="20" rx="3" stroke="#00E5C8" strokeWidth="1.5" />
              <circle cx="18" cy="18" r="5" stroke="#00E5C8" strokeWidth="1.5" />
              <circle cx="18" cy="18" r="2" fill="#00E5C8" />
              <rect x="24" y="10" width="4" height="3" rx="1" fill="#00E5C840" />
            </svg>
          </div>
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 15, color: "#E8F8FF", lineHeight: 1.3 }}>
            Guided Skin<br />Lesion Imaging
          </div>
          <div style={{ fontFamily: "system-ui", fontSize: 10, color: "#5B8FA8", marginTop: 5, lineHeight: 1.4 }}>
            Touch-guided · AI-powered · 60 seconds
          </div>
        </div>
      </div>

      {/* Start button */}
      <button onClick={onNext} style={{
        ...S.bigBtn,
        background: "linear-gradient(135deg, #00C9AD, #00E5C8)",
        boxShadow: "0 4px 20px #00E5C830",
        color: "#021A24",
        fontSize: 16,
        fontWeight: 800,
        letterSpacing: "0.05em",
      }}>
        ▶ &nbsp; TAP TO BEGIN
      </button>
    </div>
  );
}

// ─── SCREEN 2: Prep Instructions ─────────────────────────────────────────────
function PrepScreen({ onNext, onBack }) {
  const steps = [
    { icon: "💡", text: "Find good lighting or stay near the device" },
    { icon: "🧴", text: "Clean the skin area gently" },
    { icon: "📏", text: "Keep the lesion centered & uncovered" },
  ];
  return (
    <div style={{ ...S.screen, background: "#061B28", padding: "16px 20px 16px" }}>
      <div style={{ ...S.screenTitle }}>Before We Start</div>
      <StepDots total={4} current={1} />

      <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 10, flex: 1 }}>
        {steps.map((s, i) => (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: 12,
            background: "#0A2535",
            border: "1px solid #1A3D52",
            borderRadius: 10, padding: "10px 14px",
            animation: `slideIn 0.3s ease ${i * 0.1}s both`,
          }}>
            <div style={{ fontSize: 22, minWidth: 30, textAlign: "center" }}>{s.icon}</div>
            <div style={{ fontFamily: "system-ui", fontSize: 12, color: "#B0D4E8", lineHeight: 1.4 }}>{s.text}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
        <button onClick={onBack} style={{ ...S.ghostBtn, flex: 1 }}>← Back</button>
        <button onClick={onNext} style={{ ...S.primaryBtn, flex: 2 }}>I'm Ready →</button>
      </div>
    </div>
  );
}

// ─── SCREEN 3: Positioning Guide ──────────────────────────────────────────────
function PositionScreen({ onNext, onBack }) {
  const [step, setStep] = useState(0);
  const guides = [
    { label: "Hold arm", icon: "🖐", hint: "Extend your arm and hold skin taut", color: "#FFD32A" },
    { label: "Distance", icon: "📐", hint: "Keep 5–10 cm from the camera lens", color: "#FF6B81" },
    { label: "Center it", icon: "🎯", hint: "Align the lesion inside the circle", color: "#00E5C8" },
  ];
  const g = guides[step];
  return (
    <div style={{ ...S.screen, background: "#061B28", padding: "14px 20px" }}>
      <div style={{ ...S.screenTitle }}>Position Guide</div>
      <StepDots total={4} current={2} />

      <div style={{ display: "flex", gap: 14, alignItems: "stretch", marginTop: 10, flex: 1 }}>
        {/* Left: visual guide */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, flex: 1 }}>
          <div style={{ position: "relative" }}>
            <Viewfinder status="idle" />
            {/* Animated hand/arrow */}
            <div style={{
              position: "absolute",
              bottom: -18, left: "50%", transform: "translateX(-50%)",
              fontSize: 28,
              animation: "bounce 1.2s ease-in-out infinite",
            }}>{g.icon}</div>
          </div>
          <div style={{
            marginTop: 20,
            background: `${g.color}15`,
            border: `1px solid ${g.color}40`,
            borderRadius: 8,
            padding: "6px 12px",
            fontFamily: "system-ui", fontSize: 10, color: g.color, textAlign: "center",
          }}>{g.hint}</div>
        </div>

        {/* Right: step selector */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8, justifyContent: "center" }}>
          {guides.map((gd, i) => (
            <button key={i} onClick={() => setStep(i)} style={{
              background: i === step ? `${gd.color}20` : "#0A2535",
              border: `1.5px solid ${i === step ? gd.color : "#1A3D52"}`,
              borderRadius: 8, padding: "8px 10px",
              color: i === step ? gd.color : "#4A7A8A",
              fontFamily: "system-ui", fontSize: 11,
              cursor: "pointer", textAlign: "left",
              minWidth: 80,
              transition: "all 0.2s",
            }}>
              <div style={{ fontSize: 16 }}>{gd.icon}</div>
              <div style={{ fontSize: 10, marginTop: 2 }}>{gd.label}</div>
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
        <button onClick={onBack} style={{ ...S.ghostBtn, flex: 1 }}>← Back</button>
        <button onClick={onNext} style={{ ...S.primaryBtn, flex: 2 }}>Open Camera →</button>
      </div>
    </div>
  );
}

// ─── SCREEN 4: Capture ───────────────────────────────────────────────────────
function CaptureScreen({ onNext, onBack }) {
  const [status, setStatus] = useState("idle"); // idle | good | bad | capturing
  const [countdown, setCountdown] = useState(null);
  const [feedback, setFeedback] = useState("Align the lesion inside the target");
  const timerRef = useRef();

  const simulate = () => {
    // Simulate detection feedback cycle
    setStatus("bad");
    setFeedback("Move closer to camera");
    setTimeout(() => {
      setStatus("good");
      setFeedback("✓ Perfect position! Hold still...");
    }, 1500);
    setTimeout(() => {
      setStatus("capturing");
      setFeedback("📸 Capturing...");
      let c = 3;
      setCountdown(c);
      timerRef.current = setInterval(() => {
        c--;
        if (c <= 0) {
          clearInterval(timerRef.current);
          setCountdown(null);
          onNext();
        } else {
          setCountdown(c);
        }
      }, 1000);
    }, 3200);
  };

  useEffect(() => () => clearInterval(timerRef.current), []);

  const statusColor = { idle: "#00E5C8", good: "#00FF88", bad: "#FF4757", capturing: "#FFD32A" };
  const c = statusColor[status];

  return (
    <div style={{ ...S.screen, background: "#030E15", padding: "12px 16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div style={{ fontFamily: "monospace", fontSize: 11, color: "#00E5C8", letterSpacing: "0.1em" }}>SkinSense AI</div>
        <StepDots total={4} current={3} />
        <div style={{
          width: 10, height: 10, borderRadius: "50%",
          background: c,
          boxShadow: `0 0 8px ${c}`,
          animation: status === "capturing" ? "blink 0.5s infinite" : "none",
        }} />
      </div>

      {/* Camera feed simulation */}
      <div style={{
        flex: 1,
        background: "linear-gradient(135deg, #0A1A20 0%, #051015 100%)",
        borderRadius: 10,
        border: `1.5px solid ${c}40`,
        position: "relative",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
        minHeight: 140,
      }}>
        {/* Simulated skin texture */}
        <div style={{
          position: "absolute", inset: 0,
          background: "radial-gradient(ellipse at 50% 50%, #8B6045 0%, #6B4830 40%, #051015 100%)",
          opacity: status === "idle" ? 0 : status === "bad" ? 0.3 : 0.6,
          transition: "opacity 0.5s",
        }} />

        {/* Scanline effect */}
        <div style={{
          position: "absolute", inset: 0,
          backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, #00000015 2px, #00000015 4px)",
          pointerEvents: "none",
        }} />

        {/* Viewfinder */}
        <div style={{ position: "relative", zIndex: 2 }}>
          <Viewfinder status={status} />
        </div>

        {/* Countdown overlay */}
        {countdown !== null && (
          <div style={{
            position: "absolute", inset: 0, display: "flex",
            alignItems: "center", justifyContent: "center",
            zIndex: 10,
          }}>
            <div style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: 72, fontWeight: 900,
              color: "#FFD32A",
              textShadow: "0 0 30px #FFD32A80",
              animation: "countPulse 1s ease-in-out infinite",
            }}>{countdown}</div>
          </div>
        )}

        {/* Status bar at top of feed */}
        <div style={{
          position: "absolute", top: 8, left: 8, right: 8,
          display: "flex", justifyContent: "center",
          zIndex: 5,
        }}>
          <div style={{
            background: "#00000070",
            borderRadius: 20,
            padding: "4px 12px",
            fontFamily: "system-ui",
            fontSize: 10,
            color: c,
            backdropFilter: "blur(4px)",
          }}>{feedback}</div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <button onClick={onBack} style={{ ...S.ghostBtn, flex: 1, fontSize: 11 }}>← Back</button>
        <button onClick={simulate} disabled={status !== "idle"} style={{
          ...S.primaryBtn, flex: 2,
          background: status !== "idle"
            ? `linear-gradient(135deg, ${c}40, ${c}60)`
            : "linear-gradient(135deg, #00C9AD, #00E5C8)",
          color: status !== "idle" ? c : "#021A24",
          fontSize: 13,
        }}>
          {status === "idle" ? "📸  Start Scan" : status === "capturing" ? "Capturing..." : status === "good" ? "✓ Hold Still" : "⚠ Adjust Position"}
        </button>
      </div>
    </div>
  );
}

// ─── SCREEN 5: Analyzing ──────────────────────────────────────────────────────
function AnalyzingScreen({ onNext }) {
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState(0);
  const stages = ["Preprocessing image...", "Detecting lesion boundaries...", "Analyzing pigmentation...", "Running AI model...", "Generating report..."];

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((p) => {
        const next = p + 2;
        setStage(Math.min(Math.floor((next / 100) * stages.length), stages.length - 1));
        if (next >= 100) {
          clearInterval(interval);
          setTimeout(onNext, 600);
        }
        return Math.min(next, 100);
      });
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ ...S.screen, background: "#061B28", justifyContent: "center", alignItems: "center", padding: 24 }}>
      {/* Scanning animation */}
      <div style={{ position: "relative", width: 120, height: 120, display: "flex", alignItems: "center", justifyContent: "center" }}>
        {/* Rotating ring */}
        <div style={{
          position: "absolute", width: 110, height: 110,
          borderRadius: "50%",
          border: "2px solid transparent",
          borderTopColor: "#00E5C8",
          borderRightColor: "#00E5C840",
          animation: "spin 1.2s linear infinite",
        }} />
        <div style={{
          position: "absolute", width: 90, height: 90,
          borderRadius: "50%",
          border: "1.5px solid transparent",
          borderBottomColor: "#00A89050",
          animation: "spin 2s linear infinite reverse",
        }} />
        {/* Center */}
        <div style={{
          width: 64, height: 64, borderRadius: "50%",
          background: "linear-gradient(135deg, #0E3A50, #061B28)",
          border: "1px solid #00E5C830",
          display: "flex", alignItems: "center", justifyContent: "center",
          flexDirection: "column", gap: 2,
        }}>
          <div style={{ fontFamily: "monospace", fontSize: 16, fontWeight: 700, color: "#00E5C8" }}>
            {progress}%
          </div>
          <div style={{ fontFamily: "monospace", fontSize: 7, color: "#00E5C860", letterSpacing: "0.1em" }}>AI</div>
        </div>
      </div>

      <div style={{ marginTop: 20, fontFamily: "'DM Serif Display', serif", fontSize: 15, color: "#E8F8FF", textAlign: "center" }}>
        Analyzing your scan
      </div>

      {/* Stage label */}
      <div style={{
        marginTop: 8,
        fontFamily: "monospace", fontSize: 10,
        color: "#00E5C880", textAlign: "center",
        minHeight: 16,
        letterSpacing: "0.05em",
      }}>
        {stages[stage]}
      </div>

      {/* Progress bar */}
      <div style={{ width: "100%", height: 4, background: "#0A2535", borderRadius: 2, marginTop: 16, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${progress}%`,
          background: "linear-gradient(90deg, #00C9AD, #00E5C8)",
          borderRadius: 2,
          transition: "width 0.1s linear",
          boxShadow: "0 0 8px #00E5C860",
        }} />
      </div>

      <div style={{ marginTop: 10, fontFamily: "system-ui", fontSize: 10, color: "#3A6A80" }}>
        Please wait — do not move the device
      </div>
    </div>
  );
}

// ─── SCREEN 6: Result ─────────────────────────────────────────────────────────
function ResultScreen({ onRestart }) {
  const [revealed, setRevealed] = useState(false);
  useEffect(() => { setTimeout(() => setRevealed(true), 200); }, []);

  const metrics = [
    { label: "Asymmetry", value: "Low", score: 15, color: "#00FF88" },
    { label: "Border", value: "Regular", score: 20, color: "#00FF88" },
    { label: "Color", value: "Uniform", score: 12, color: "#00FF88" },
    { label: "Diameter", value: "4.2 mm", score: 42, color: "#FFD32A" },
  ];

  return (
    <div style={{ ...S.screen, background: "#061B28", padding: "14px 18px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: "#00E5C8" }}>Analysis Report</div>
        <div style={{ fontFamily: "monospace", fontSize: 9, color: "#3A6A80" }}>{new Date().toLocaleDateString()}</div>
      </div>

      {/* Risk indicator */}
      <div style={{
        marginTop: 10,
        background: "#00FF8815",
        border: "1.5px solid #00FF8840",
        borderRadius: 12,
        padding: "10px 16px",
        display: "flex",
        alignItems: "center",
        gap: 12,
        animation: revealed ? "slideIn 0.4s ease" : "none",
      }}>
        <div style={{ fontSize: 32 }}>🟢</div>
        <div>
          <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 16, color: "#00FF88" }}>Low Risk</div>
          <div style={{ fontFamily: "system-ui", fontSize: 10, color: "#5BAA70", lineHeight: 1.4 }}>
            No urgent concern detected. Monitor monthly.
          </div>
        </div>
        <div style={{ marginLeft: "auto", fontFamily: "monospace", fontSize: 22, fontWeight: 700, color: "#00FF88" }}>
          92<span style={{ fontSize: 11 }}>/100</span>
        </div>
      </div>

      {/* ABCD Metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginTop: 8 }}>
        {metrics.map((m, i) => (
          <div key={i} style={{
            background: "#0A2535", borderRadius: 8,
            padding: "8px 10px",
            border: "1px solid #1A3D52",
            animation: revealed ? `slideIn 0.3s ease ${0.1 + i * 0.08}s both` : "none",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
              <span style={{ fontFamily: "system-ui", fontSize: 10, color: "#5B8FA8" }}>{m.label}</span>
              <span style={{ fontFamily: "monospace", fontSize: 10, color: m.color }}>{m.value}</span>
            </div>
            <div style={{ height: 3, background: "#1A3D52", borderRadius: 2, overflow: "hidden" }}>
              <div style={{
                height: "100%", width: revealed ? `${m.score}%` : "0%",
                background: m.color, borderRadius: 2,
                transition: `width 0.8s ease ${0.3 + i * 0.1}s`,
                boxShadow: `0 0 6px ${m.color}60`,
              }} />
            </div>
          </div>
        ))}
      </div>

      {/* Disclaimer */}
      <div style={{
        marginTop: 8,
        background: "#FFD32A10", borderRadius: 6,
        padding: "6px 10px",
        fontFamily: "system-ui", fontSize: 9,
        color: "#8A7030", lineHeight: 1.4,
        border: "1px solid #FFD32A20",
      }}>
        ⚠ This is a screening aid only. Consult a dermatologist for diagnosis.
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
        <button style={{ ...S.ghostBtn, flex: 1, fontSize: 10 }}>📋 Save Report</button>
        <button onClick={onRestart} style={{ ...S.primaryBtn, flex: 1, fontSize: 11 }}>🔄 New Scan</button>
      </div>
    </div>
  );
}

// ─── Shared styles ─────────────────────────────────────────────────────────────
const S = {
  screen: {
    width: 480, height: 272,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    fontFamily: "system-ui, sans-serif",
  },
  screenTitle: {
    fontFamily: "'DM Mono', monospace",
    fontSize: 13,
    color: "#E8F8FF",
    fontWeight: 600,
    letterSpacing: "0.05em",
    marginBottom: 8,
  },
  bigBtn: {
    height: 44,
    borderRadius: 10,
    border: "none",
    cursor: "pointer",
    fontFamily: "'DM Mono', monospace",
    display: "flex", alignItems: "center", justifyContent: "center",
    transition: "transform 0.1s, opacity 0.2s",
  },
  primaryBtn: {
    height: 38,
    borderRadius: 8,
    border: "none",
    cursor: "pointer",
    fontFamily: "system-ui",
    fontSize: 12,
    fontWeight: 700,
    background: "linear-gradient(135deg, #00C9AD, #00E5C8)",
    color: "#021A24",
    display: "flex", alignItems: "center", justifyContent: "center",
    letterSpacing: "0.02em",
    transition: "opacity 0.15s",
  },
  ghostBtn: {
    height: 38,
    borderRadius: 8,
    border: "1.5px solid #1A3D52",
    cursor: "pointer",
    fontFamily: "system-ui",
    fontSize: 11,
    color: "#5B8FA8",
    background: "transparent",
    display: "flex", alignItems: "center", justifyContent: "center",
    transition: "border-color 0.15s, color 0.15s",
  },
  logoMark: {
    width: 38, height: 38,
    background: "#00E5C815",
    border: "1px solid #00E5C830",
    borderRadius: 10,
    display: "flex", alignItems: "center", justifyContent: "center",
  },
};

// ─── Root App ─────────────────────────────────────────────────────────────────
export default function SkinSenseApp() {
  const [screen, setScreen] = useState(SCREENS.WELCOME);
  const [showDevice, setShowDevice] = useState(true);

  const go = (s) => setScreen(s);

  const screenMap = {
    [SCREENS.WELCOME]: <WelcomeScreen onNext={() => go(SCREENS.PREP)} />,
    [SCREENS.PREP]: <PrepScreen onNext={() => go(SCREENS.POSITION)} onBack={() => go(SCREENS.WELCOME)} />,
    [SCREENS.POSITION]: <PositionScreen onNext={() => go(SCREENS.CAPTURE)} onBack={() => go(SCREENS.PREP)} />,
    [SCREENS.CAPTURE]: <CaptureScreen onNext={() => go(SCREENS.ANALYZING)} onBack={() => go(SCREENS.POSITION)} />,
    [SCREENS.ANALYZING]: <AnalyzingScreen onNext={() => go(SCREENS.RESULT)} />,
    [SCREENS.RESULT]: <ResultScreen onRestart={() => go(SCREENS.WELCOME)} />,
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#010B12",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: 20,
      fontFamily: "system-ui",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;700&family=DM+Serif+Display&display=swap');

        @keyframes pulseRing {
          0% { transform: scale(0.6); opacity: 0.6; }
          100% { transform: scale(1.2); opacity: 0; }
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes bounce {
          0%, 100% { transform: translateX(-50%) translateY(0); }
          50% { transform: translateX(-50%) translateY(-6px); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes countPulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.1); }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.2; }
        }
        button:active { opacity: 0.75; transform: scale(0.98); }
      `}</style>

      {/* Label */}
      <div style={{ marginBottom: 16, textAlign: "center" }}>
        <div style={{ fontFamily: "monospace", fontSize: 11, color: "#2A5A70", letterSpacing: "0.15em", textTransform: "uppercase" }}>
          SkinSense AI · Nextion NX4827T043 · 480 × 272px UI Prototype
        </div>
      </div>

      {/* Device frame */}
      <div style={{
        position: "relative",
        padding: showDevice ? "14px 18px" : 0,
        background: showDevice ? "linear-gradient(180deg, #1A2830 0%, #111D24 100%)" : "transparent",
        borderRadius: showDevice ? 14 : 0,
        boxShadow: showDevice ? "0 0 0 2px #243A46, 0 8px 40px #00000080, 0 0 60px #00E5C808" : "none",
      }}>
        {/* Status LEDs (decorative) */}
        {showDevice && (
          <div style={{ position: "absolute", top: 6, right: 24, display: "flex", gap: 5 }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#00FF88", boxShadow: "0 0 4px #00FF88" }} />
            <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#0A2535" }} />
          </div>
        )}

        {/* Screen content */}
        <div style={{ width: 480, height: 272, borderRadius: showDevice ? 4 : 0, overflow: "hidden" }}>
          {screenMap[screen]}
        </div>

        {/* Bottom bezel label */}
        {showDevice && (
          <div style={{
            textAlign: "center", marginTop: 8,
            fontFamily: "monospace", fontSize: 8,
            color: "#2A4A5A", letterSpacing: "0.2em",
          }}>
            NEXTION NX4827T043 · 4.3" TFT TOUCH
          </div>
        )}
      </div>

      {/* Controls */}
      <div style={{ marginTop: 20, display: "flex", gap: 12, alignItems: "center" }}>
        <div style={{ fontFamily: "monospace", fontSize: 10, color: "#2A5A70" }}>Current screen:</div>
        {Object.values(SCREENS).map((s) => (
          <button key={s} onClick={() => go(s)} style={{
            background: screen === s ? "#00E5C820" : "transparent",
            border: `1px solid ${screen === s ? "#00E5C840" : "#1A3D52"}`,
            borderRadius: 5, padding: "4px 8px",
            fontFamily: "monospace", fontSize: 9,
            color: screen === s ? "#00E5C8" : "#3A6A80",
            cursor: "pointer", textTransform: "uppercase", letterSpacing: "0.1em",
          }}>{s}</button>
        ))}
        <button onClick={() => setShowDevice(!showDevice)} style={{
          background: "transparent",
          border: "1px solid #1A3D52",
          borderRadius: 5, padding: "4px 8px",
          fontFamily: "monospace", fontSize: 9,
          color: "#3A6A80", cursor: "pointer",
        }}>
          {showDevice ? "Hide Frame" : "Show Frame"}
        </button>
      </div>
    </div>
  );
}