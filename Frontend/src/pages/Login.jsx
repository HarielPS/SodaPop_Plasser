import { useState } from "react";
import "./login.css";

export default function Login({ onLogin }) {
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();

    if (user.trim() !== "" && pass.trim() !== "") {
      onLogin();
    } else {
      alert("Please, enter your credentials");
    }
  };

  return (
    <div className="login-container">
      <div className="login-box">

        <img src="/logoaustria.png" className="logo-img" alt="Fast Tamping AI Logo" />

        <h2 className="title">Fast Tamping AI</h2>
        <p className="subtitle">Railway Maintenance</p>

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