import { useState } from "react";
import { UserPlus, Lock } from "lucide-react";

// Componente para la entrada de datos con el estilo oscuro
const StyledInput = ({ label, name, value, onChange, type = "text", required = false, min, max }) => (
    <div>
        <label className="block text-sm font-medium text-gray-300">{label}</label>
        <input
            type={type}
            name={name}
            value={value}
            onChange={onChange}
            required={required}
            min={min}
            max={max}
            // Estilos oscuros para el input
            className="mt-1 block w-full p-3 rounded-lg shadow-inner bg-neutral-700 text-white border border-neutral-600 focus:ring-yellow-500 focus:border-yellow-500 transition duration-150"
        />
    </div>
);

export default function UserRegistration() {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    firstName: "",
    lastName: "",
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setMessage(null);
    
    console.log("Datos de Usuario a Registrar:", formData);

    setTimeout(() => {
      setIsSubmitting(false);
      setMessage({
        type: "success",
        text: `Usuario "${formData.username}" registrado con éxito.`,
      });
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-neutral-900 p-8 flex justify-center">
      <div className="w-full max-w-xl bg-neutral-800 p-8 rounded-xl border-2 border-yellow-500 shadow-xl shadow-yellow-500/10">
        <div className="flex items-center space-x-3 mb-6 border-b border-neutral-700 pb-4">
          <UserPlus className="w-8 h-8 text-yellow-500" />
          <h2 className="text-2xl font-bold text-white">
            Registro de Nuevo Usuario
          </h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <StyledInput label="Nombre de Usuario" name="username" value={formData.username} onChange={handleChange} required />
            <StyledInput label="Email" name="email" value={formData.email} onChange={handleChange} type="email" required />
            <StyledInput label="Nombre" name="firstName" value={formData.firstName} onChange={handleChange} />
            <StyledInput label="Apellido" name="lastName" value={formData.lastName} onChange={handleChange} />
          </div>

          <div className="flex items-center space-x-2">
            <Lock className="w-5 h-5 text-gray-400" />
            <StyledInput label="Contraseña" name="password" value={formData.password} onChange={handleChange} type="password" required />
          </div>

          {message && (
            <div className={`p-3 rounded-lg text-sm ${message.type === 'success' ? 'bg-green-700 text-white' : 'bg-red-700 text-white'}`}>
              {message.text}
            </div>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className={`w-full py-3 px-4 border border-transparent rounded-lg shadow-lg text-lg font-bold text-black transition duration-150 ${
              isSubmitting ? 'bg-yellow-400 cursor-not-allowed' : 'bg-yellow-500 hover:bg-yellow-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500'
            }`}
          >
            {isSubmitting ? "Registrando..." : "Registrar Usuario"}
          </button>
        </form>
      </div>
    </div>
  );
}