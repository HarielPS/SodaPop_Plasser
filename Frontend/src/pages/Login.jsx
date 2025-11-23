import { useState } from "react";
import "./Login.css";

export default function Login({ onLogin }) {
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    const username = user.trim();
    const password = pass.trim();

    //Credenciales del administrador
    const adminUser = "admin";
    const adminPass = "Plasser2025!";

    if (username === adminUser && password === adminPass) {
      onLogin("admin", "Admin");
      return;
    }

    // Lista de operadores guardados
    const operators = JSON.parse(localStorage.getItem("operators")) || [];

    const found = operators.find(
      (op) => op.username === username && op.password === password
    );

    if (found) {
      const fullName = `${found.name} ${found.lastname}`;
      onLogin("operator", fullName);
      return;
    }

    setError("Incorrect username or password");
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <img
          src="/logoaustria.png"
          className="logo-img"
          alt="Fast Tamping AI Logo"
        />

        <h2 className="title">Fast Tamping AI</h2>
        <p className="subtitle">Railway Maintenance</p>

        {error && <p className="error-msg">{error}</p>}

        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Username"
            value={user}
            onChange={(e) => setUser(e.target.value)}
          />

          <input
            type="password"
            placeholder="Password"
            value={pass}
            onChange={(e) => setPass(e.target.value)}
          />

          <button type="submit">Login</button>
        </form>
      </div>
    </div>
  );
}