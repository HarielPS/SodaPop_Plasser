import { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";

import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import AdminPanel from "./pages/AdminPanel";
import DiagnosticsHistory from "./pages/DiagnosticsHistory";

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

  // Si no hay rol, siempre login
  if (!role) return <Login onLogin={handleLogin} />;

  return (
    <Routes>
      {/* ADMIN */}
      {role === "admin" && (
        <>
          <Route
            path="/admin"
            element={<AdminPanel onLogout={handleLogout} />}
          />
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
            path="/diagnostics"
            element={<DiagnosticsHistory user={currentUser} />}
          />

          <Route path="*" element={<Navigate to="/dashboard" />} />
        </>
      )}
    </Routes>
  );
}