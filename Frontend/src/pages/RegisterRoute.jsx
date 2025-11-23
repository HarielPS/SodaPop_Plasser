import { useState } from "react";
import Navbar from "../components/Navbar";
import "./AdminPanel.css";

export default function RegisterRoute() {
  const [routeCode, setRouteCode] = useState("");
  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!routeCode || !origin || !destination) {
      alert("All fields are required");
      return;
    }

    const newRoute = { routeCode, origin, destination };
    const saved = JSON.parse(localStorage.getItem("routes")) || [];
    localStorage.setItem("routes", JSON.stringify([...saved, newRoute]));

    setRouteCode(""); 
    setOrigin(""); 
    setDestination("");
    alert("Route registered!");
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
        <h3>Register Route</h3>
        <form onSubmit={handleSubmit}>
          <input
            placeholder="Route Code"
            value={routeCode}
            onChange={e => setRouteCode(e.target.value)}
          />
          <input
            placeholder="Origin Station"
            value={origin}
            onChange={e => setOrigin(e.target.value)}
          />
          <input
            placeholder="Destination Station"
            value={destination}
            onChange={e => setDestination(e.target.value)}
          />
          <button type="submit" className="btn-add">Save Route</button>
        </form>
      </div>
    </div>
  );
}
