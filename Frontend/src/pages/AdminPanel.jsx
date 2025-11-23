
import { useState } from "react";
import "./AdminPanel.css";

export default function AdminPanel({ onLogout }) {
  const [name, setName] = useState("");
  const [lastname, setLastname] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const [operators, setOperators] = useState(() => {
    return JSON.parse(localStorage.getItem("operators")) || [];
  });

  // === SAVE TO LOCALSTORAGE ===
  const saveOperators = (list) => {
    localStorage.setItem("operators", JSON.stringify(list));
    setOperators(list);
  };

  // === ADD NEW OPERATOR ===
  const handleAdd = (e) => {
    e.preventDefault();

    if (
      !name.trim() ||
      !lastname.trim() ||
      !username.trim() ||
      !password.trim()
    ) {
      alert("All fields are required");
      return;
    }

    const newOperator = { name, lastname, username, password };

    saveOperators([...operators, newOperator]);

    setName("");
    setLastname("");
    setUsername("");
    setPassword("");
  };

  return (
    <div className="admin-container">
      {/* Header with logout */}
      <div className="admin-header">
        <h1 className="admin-title">Admin Panel</h1>
        <button className="logout-btn" onClick={onLogout}>
          Log Out
        </button>
      </div>

      {/* Form to add operator */}
      <form className="admin-form" onSubmit={handleAdd}>
        <input
          type="text"
          placeholder="First Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />

        <input
          type="text"
          placeholder="Last Name"
          value={lastname}
          onChange={(e) => setLastname(e.target.value)}
        />

        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button type="submit" className="add-btn">
          Add Operator
        </button>
      </form>

      {/* Operators list */}
      <h2 className="list-title">Registered Operators</h2>

      <ul className="operator-list">
        {operators.length === 0 && (
          <p className="no-ops">No operators registered yet.</p>
        )}

        {operators.map((op, index) => (
          <li key={index} className="operator-item">
            <strong>
              {op.name} {op.lastname}
            </strong>{" "}
            — {op.username}
          </li>
        ))}
      </ul>
    </div>
  );
}
