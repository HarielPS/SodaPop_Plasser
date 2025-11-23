import { Link } from "react-router-dom";
import "./Navbar.css";

export default function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/admin/register-user">RegisterUser</Link>
      <Link to="/admin/register-route">RegisterRoute</Link>
      <Link to="/admin/register-train">RegisterTrain</Link>
      <Link to="/admin/register-machine">RegisterMachine</Link>
      <Link to="/admin/register-traintrip">RegisterTraintrip</Link>
    </nav>
  );
}
