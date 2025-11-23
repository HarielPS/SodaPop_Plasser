import { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import RegisterRoute from "./pages/RegisterRoute";
import RegisterMachine from "./pages/RegisterMachine";
import RegisterTrain from "./pages/RegisterTrain";
import RegisterTraintrip from "./pages/RegisterTraintrip";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import AdminPanel from "./pages/AdminPanel";
import DiagnosticsHistory from "./pages/DiagnosticsHistory";
import TrainCameras from "./pages/MachineCameras";

export default function App() {
  const [role, setRole] = useState(null);
  const [currentUser, setCurrentUser] = useState(null);

  const handleLogin = (role, userData) => {
    setRole(role);
    setCurrentUser(userData);
  };

  const handleLogout = () => {
    setRole(null);
    setCurrentUser(null);
  };

  if (!role) return <Login onLogin={handleLogin} />;

  return (
    <Routes>
      {/* ADMIN */}
      {role === "admin" && (
        <>
          {/* Panel principal */}
          <Route
            path="/admin"
            element={<AdminPanel onLogout={handleLogout} />}
          />

          {/* Formulario RegisterRoute */}
          <Route
            path="/admin/register-route"
            element={<RegisterRoute />}
          />
          <Route
            path="/admin/register-train"
            element={<RegisterTrain />}
          />
          <Route
            path="/admin/register-machine"
            element={<RegisterMachine />}
          />
          <Route
            path="/admin/register-traintrip"
            element={<RegisterTraintrip />}
          />  

          {/* Cualquier otra ruta redirige al panel */}
          <Route path="*" element={<Navigate to="/admin" />} />
        </>
      )}

      {/* OPERATOR */}
      {role === "operator" && (
        <>
          <Route
            path="/dashboard"
            element={<Dashboard user={currentUser} />}
          />

          <Route
            path="/train-cameras"
            element={<TrainCameras user={currentUser} />}
          />

          <Route
            path="/diagnostics"
            element={<DiagnosticsHistory user={currentUser} />}
          />

          <Route path="*" element={<Navigate to="/dashboard" />} />
        </>
      )}
    </Routes>
  );
}
