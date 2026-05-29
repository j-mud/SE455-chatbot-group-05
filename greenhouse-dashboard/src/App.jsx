import { useEffect, useState } from "react"
import "./App.css"


//most Common Plant options

const commonPlants = [
  "Tomato",
  "Mint",
  "Basil",
  "Rosemary",
  "Aloe Vera",
  "Date Palm",
  "Lemon Tree",
  "Cucumber",
  "Pepper",
  "Jasmine",
  "Bougainvillea",
  "Lavender",
]

function App() {
 
  // Sensor data state
  
  const [data, setData] = useState({
    soil: "0",
    temp: "0",
    humidity: "0",
    gas: "0",
    status: "WAITING",
    pump: "OFF",
    ai_message: "Awaiting sensor readings...",
  })

 
  // User Input States
 
  const [plantName, setPlantName] = useState("Tomato")
  const [customPlant, setCustomPlant] = useState("")
  const [history, setHistory] = useState([])
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState(
    "Ask about current conditions or historical data — e.g. 'What was the humidity like one hour ago?'"
  )

 
  // Fetch Live Sensor Data
 
  async function fetchData() {
    try {
      const response = await fetch("http://127.0.0.1:5000/sensor-data")
      const result = await response.json()

      setData(result)

      setHistory((prev) => [
        {
          time: new Date().toLocaleTimeString(),
          ...result,
        },
        ...prev.slice(0, 7),
      ])
    } catch (error) {
      console.error("Failed to fetch sensor data:", error)
    }
  }

  
  // Auto Refresh Sensor Data
  
  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 2000)
    return () => clearInterval(interval)
  }, [])

 
  // Plant Selection
 
  const activePlant = customPlant.trim() !== "" ? customPlant : plantName

 
  // Command SLM Logic

async function analyzeConditions() {
  try {
    const response = await fetch("http://127.0.0.1:5000/ask-command", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
      }),
    })

    const result = await response.json()

    setAnswer(result.answer)

  } catch (error) {
    console.error("Command SLM request failed:", error)

    setAnswer(
      "Command SLM could not process the request. Check the local backend connection."
    )
  }
}

  
  // Dashboard Sensor widget
  
  const cards = [
    ["Temperature", `${data.temp} °C`],
    ["Humidity", `${data.humidity} %`],
    ["Soil Moisture", data.soil],
    ["Gas Level", data.gas],
    ["Pump Status", data.pump],
    ["System Status", data.status],
  ]

  return (
    <div
      style={{
        minHeight: "100vh",
        padding: "35px",
        fontFamily: "Arial, sans-serif",
        background: "linear-gradient(135deg, #f8f5ef, #edf7ee, #fdfaf4)",
      }}
    >
      {/* Main Dashboard Container */}
      <div
        style={{
          background: "rgba(255,255,255,0.96)",
          borderRadius: "20px",
          padding: "30px",
          boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
        }}
      >
        {/* Dashboard Header */}
        <h1
          style={{
            textAlign: "center",
            color: "#1b5e20",
            fontSize: "42px",
            marginBottom: "10px",
          }}
        >
          Smart Irrigation Monitoring Dashboard
        </h1>

        <p
          style={{
            textAlign: "center",
            color: "#556b5d",
            fontSize: "18px",
            marginBottom: "30px",
          }}
        >
          Live greenhouse monitoring powered by Arduino Nano ESP32 and edge
          intelligence.
        </p>

        {/* Plant Selection Section */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: "15px",
            flexWrap: "wrap",
            marginTop: "20px",
            padding: "20px",
            borderRadius: "18px",
            background: "#f5faf4",
            border: "1px solid #d7ead7",
          }}
        >
          <label
            style={{
              fontWeight: "bold",
              color: "#2e5f3e",
              fontSize: "13px",
              alignSelf: "center",
            }}
          >
            Choose plant:
          </label>

          <select
            value={plantName}
            onChange={(e) => setPlantName(e.target.value)}
            style={{
              padding: "12px",
              borderRadius: "12px",
              border: "1px solid #b7d6b0",
              background: "white",
              color: "#2e5f3e",
              fontSize: "13px",
            }}
          >
            {commonPlants.map((plant) => (
              <option key={plant}>{plant}</option>
            ))}
          </select>

          <input
            placeholder="Or type plant name"
            value={customPlant}
            onChange={(e) => setCustomPlant(e.target.value)}
            style={{
              padding: "12px",
              borderRadius: "12px",
              border: "1px solid #b7d6b0",
              background: "white",
              color: "#2e5f3e",
              width: "250px",
              fontSize: "13px",
            }}
          />
        </div>

        {/* Active Plant Display */}
        <h2
          style={{
            textAlign: "center",
            marginTop: "25px",
            color: "#2e7d32",
            fontSize: "28px",
          }}
        >
          Currently Monitoring: {activePlant}
        </h2>

        {/* Live Sensor Monitoring Cards */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
            gap: "22px",
            marginTop: "35px",
          }}
        >
          {cards.map((card, index) => (
            <div
              key={index}
              style={{
                background: "#ffffff",
                borderRadius: "18px",
                padding: "26px",
                textAlign: "center",
                border: "1px solid #d7ead7",
                boxShadow: "0 6px 16px rgba(0,0,0,0.08)",
              }}
            >
              <h3
                style={{
                  color: "#2e7d32",
                  fontSize: "20px",
                  marginBottom: "12px",
                }}
              >
                {card[0]}
              </h3>

              <h2
                style={{
                  color: "#44544a",
                  fontSize: "20px",
                  margin: 0,
                }}
              >
                {card[1]}
              </h2>
            </div>
          ))}
        </div>

        {/* Edge Intelligence Recommendation Section */}
        <div
          style={{
            marginTop: "35px",
            background: "linear-gradient(135deg, #1b5e20, #2e7d32)",
            borderRadius: "20px",
            padding: "28px",
            color: "#f4fff4",
            textAlign: "center",
            boxShadow: "0 8px 22px rgba(0,0,0,0.16)",
          }}
        >
          <h2 style={{ fontSize: "30px", marginTop: 0 }}>
            Edge Intelligence Recommendation
          </h2>

          <p
            style={{
              fontSize: "20px",
              color: "#e7ffe7",
            }}
          >
            {data.ai_message}
          </p>

          {/* Irrigation Analysis Input */}
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              gap: "12px",
              flexWrap: "wrap",
              marginTop: "20px",
            }}
          >
            <input
              placeholder="e.g. What was the humidity one hour ago?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && analyzeConditions()}
              style={{
                padding: "14px",
                borderRadius: "12px",
                border: "none",
                width: "60%",
                minWidth: "260px",
                background: "white",
                color: "#2e5f3e",
                fontSize: "16px",
              }}
            />

            <button
              onClick={analyzeConditions}
              style={{
                padding: "14px 22px",
                borderRadius: "12px",
                border: "none",
                background: "#f4d35e",
                color: "#2e3d2f",
                fontWeight: "bold",
                cursor: "pointer",
                fontSize: "16px",
              }}
            >
              Analyze
            </button>
          </div>

          {/* Analysis Output */}
          <p
            style={{
              marginTop: "20px",
              fontSize: "18px",
              color: "#dfffdc",
              lineHeight: "1.6",
            }}
          >
            {answer}
          </p>
        </div>

        {/* Sensor History Log */}
        <div
          style={{
            marginTop: "35px",
            background: "#fffaf0",
            borderRadius: "20px",
            padding: "28px",
            border: "1px solid #f4d35e",
            color: "#556b5d",
          }}
        >
          <h2
            style={{
              textAlign: "center",
              color: "#6d5c4d",
              fontSize: "30px",
              marginTop: 0,
            }}
          >
            Sensor History Log
          </h2>

          {history.length === 0 ? (
            <p
              style={{
                textAlign: "center",
                fontSize: "17px",
                color: "#556b5d",
              }}
            >
              No sensor readings available yet.
            </p>
          ) : (
            history.map((item, index) => (
              <div
                key={index}
                style={{
                  background: "white",
                  borderRadius: "12px",
                  padding: "14px",
                  margin: "10px 0",
                  border: "1px solid #eee2b7",
                  fontSize: "16px",
                  color: "#44544a",
                }}
              >
                <strong>{item.time}</strong> — Temp: {item.temp}°C | Humidity:{" "}
                {item.humidity}% | Soil: {item.soil} | Gas: {item.gas} | Pump:{" "}
                {item.pump}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default App