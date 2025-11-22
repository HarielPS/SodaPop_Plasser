import React, { useState } from "react";
import Login from "./pages/Login.jsx";

export default function App() {
  const [logged, setLogged] = useState(false);

  return (
    <div>
      <Login onLogin={() => setLogged(true)} />
    </div>
  );
}