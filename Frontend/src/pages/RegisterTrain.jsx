import { useState } from "react";
import Navbar from "../components/Navbar";
import "./AdminPanel.css";

export default function RegisterTrain() {
  const [model, setModel] = useState("");
  const [weight, setWeight] = useState("");
  const [maxSpeed, setMaxSpeed] = useState("");
  const [trainType, setTrainType] = useState("passenger");
  const [manufacturer, setManufacturer] = useState("");
  const [yearBuilt, setYearBuilt] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!model || !weight || !maxSpeed || !trainType || !manufacturer || !yearBuilt) {
      alert("All fields are required");
      return;
    }

    const newTrain = { model, weight, maxSpeed, trainType, manufacturer, yearBuilt };
    const saved = JSON.parse(localStorage.getItem("trains")) || [];
    localStorage.setItem("trains", JSON.stringify([...saved, newTrain]));

    setModel(""); setWeight(""); setMaxSpeed(""); setTrainType("passenger"); setManufacturer(""); setYearBuilt("");
    alert("Train registered!");
  };
  
  return (
    <div className="admin-container">
      {/* Header */}
      <div className="admin-header">
        <h1 className="admin-title">Admin Panel</h1>
      </div>

      {/* Navbar */}
      <Navbar />

      {/* Form */}
      <div className="form-box">
        <h3>Register Train</h3>
        <form onSubmit={handleSubmit}>
          <input placeholder="Model" value={model} onChange={e => setModel(e.target.value)} />
          <input placeholder="Weight" type="number" value={weight} onChange={e => setWeight(e.target.value)} />
          <input placeholder="Max Speed" type="number" value={maxSpeed} onChange={e => setMaxSpeed(e.target.value)} />
          <select value={trainType} onChange={e => setTrainType(e.target.value)}>
            <option value="passenger">Passenger</option>
            <option value="cargo">Cargo</option>
            <option value="highspeed">High-Speed</option>
          </select>
          <input placeholder="Manufacturer" value={manufacturer} onChange={e => setManufacturer(e.target.value)} />
          <input placeholder="Year Built" type="number" value={yearBuilt} onChange={e => setYearBuilt(e.target.value)} />
          <button type="submit" className="btn-add">Save Train</button>
        </form>
      </div>
    </div>
  );
}
