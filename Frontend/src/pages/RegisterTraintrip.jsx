import { useState } from "react";
import Navbar from "../components/Navbar";
import "./AdminPanel.css";

export default function RegisterTraintrip() {
  const [trainId, setTrainId] = useState("");
  const [routeCode, setRouteCode] = useState("");
  const [cargoWeight, setCargoWeight] = useState("");
  const [averageSpeed, setAverageSpeed] = useState("");
  const [departureTime, setDepartureTime] = useState("");
  const [arrivalTime, setArrivalTime] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!trainId || !routeCode || !cargoWeight || !averageSpeed || !departureTime || !arrivalTime) {
      alert("All fields are required");
      return;
    }

    const newTrip = { trainId, routeCode, cargoWeight, averageSpeed, departureTime, arrivalTime };
    const saved = JSON.parse(localStorage.getItem("trips")) || [];
    localStorage.setItem("trips", JSON.stringify([...saved, newTrip]));

    setTrainId(""); setRouteCode(""); setCargoWeight(""); setAverageSpeed(""); setDepartureTime(""); setArrivalTime("");
    alert("Train trip registered!");
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
        <h3>Register Train Trip</h3>
        <form onSubmit={handleSubmit}>
          <input placeholder="Train ID" value={trainId} onChange={e => setTrainId(e.target.value)} />
          <input placeholder="Route Code" value={routeCode} onChange={e => setRouteCode(e.target.value)} />
          <input placeholder="Cargo Weight" type="number" value={cargoWeight} onChange={e => setCargoWeight(e.target.value)} />
          <input placeholder="Average Speed" type="number" value={averageSpeed} onChange={e => setAverageSpeed(e.target.value)} />
          <input placeholder="Departure Time" type="datetime-local" value={departureTime} onChange={e => setDepartureTime(e.target.value)} />
          <input placeholder="Arrival Time" type="datetime-local" value={arrivalTime} onChange={e => setArrivalTime(e.target.value)} />
          <button type="submit" className="btn-add">Save Train Trip</button>
        </form>
      </div>
    </div>
  );
}
