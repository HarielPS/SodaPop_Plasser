import { useState } from "react";
import Navbar from "../components/Navbar";
import "./AdminPanel.css";

export default function RegisterMachine() {
  const [model, setModel] = useState("");
  const [manufacturer, setManufacturer] = useState("");
  const [yearBuilt, setYearBuilt] = useState("");
  const [weight, setWeight] = useState("");
  const [maxSpeed, setMaxSpeed] = useState("");
  const [workingSpeed, setWorkingSpeed] = useState("");
  const [tamperType, setTamperType] = useState("unimat");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!model || !manufacturer || !yearBuilt || !weight || !maxSpeed || !workingSpeed || !tamperType) {
      alert("All fields are required");
      return;
    }

    const newMachine = { model, manufacturer, yearBuilt, weight, maxSpeed, workingSpeed, tamperType };
    const saved = JSON.parse(localStorage.getItem("machines")) || [];
    localStorage.setItem("machines", JSON.stringify([...saved, newMachine]));

    setModel(""); setManufacturer(""); setYearBuilt(""); setWeight(""); setMaxSpeed(""); setWorkingSpeed(""); setTamperType("unimat");
    alert("Machine registered!");
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
        <h3>Register Machine</h3>
        <form onSubmit={handleSubmit}>
          <input placeholder="Model" value={model} onChange={e => setModel(e.target.value)} />
          <input placeholder="Manufacturer" value={manufacturer} onChange={e => setManufacturer(e.target.value)} />
          <input placeholder="Year Built" type="number" value={yearBuilt} onChange={e => setYearBuilt(e.target.value)} />
          <input placeholder="Weight" type="number" value={weight} onChange={e => setWeight(e.target.value)} />
          <input placeholder="Max Speed" type="number" value={maxSpeed} onChange={e => setMaxSpeed(e.target.value)} />
          <input placeholder="Working Speed" type="number" value={workingSpeed} onChange={e => setWorkingSpeed(e.target.value)} />
          <select value={tamperType} onChange={e => setTamperType(e.target.value)}>
            <option value="unimat">Unimat</option>
            <option value="09-3x">09-3X</option>
            <option value="dynamic">Dynamic Tamper</option>
            <option value="plain_line">Plain Line Tamper</option>
          </select>
          <button type="submit" className="btn-add">Save Machine</button>
        </form>
      </div>
    </div>
  );
}
