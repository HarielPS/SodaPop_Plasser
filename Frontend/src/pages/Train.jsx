import { useState } from "react";
import { TrainIcon, Gauge } from "lucide-react";

// Opciones basadas en el TRAIN_TYPE del modelo Django
const TRAIN_TYPES = [
  { value: "passenger", label: "Pasajeros" },
  { value: "cargo", label: "Carga" },
  { value: "highspeed", label: "Alta Velocidad" },
];

const StyledInput = ({ label, name, value, onChange, type = "text", required = false, min, max, step }) => (
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
            step={step}
            // Estilos oscuros para el input
            className="mt-1 block w-full p-3 rounded-lg shadow-inner bg-neutral-700 text-white border border-neutral-600 focus:ring-yellow-500 focus:border-yellow-500 transition duration-150"
        />
    </div>
);

const StyledSelect = ({ label, name, value, onChange, required, options }) => (
    <div>
        <label className="block text-sm font-medium text-gray-300">{label}</label>
        <select
            name={name}
            value={value}
            onChange={onChange}
            required={required}
            // Estilos oscuros para el select
            className="mt-1 block w-full p-3 rounded-lg shadow-inner bg-neutral-700 text-white border border-neutral-600 focus:ring-yellow-500 focus:border-yellow-500 transition duration-150"
        >
            <option value="" disabled className="text-gray-500">Seleccione una opción</option>
            {options.map((option) => (
                <option key={option.value} value={option.value} className="bg-neutral-800 text-white">
                    {option.label}
                </option>
            ))}
        </select>
    </div>
);

export default function TrainRegistration() {
  const [formData, setFormData] = useState({
    model: "",
    manufacturer: "",
    yearBuilt: "",
    weight: "",
    maxSpeed: "",
    trainType: "",
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    const newValue = (type === 'number' || name === 'yearBuilt') ? parseFloat(value) : value;
    
    setFormData((prevData) => ({
      ...prevData,
      [name]: newValue,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setMessage(null);

    console.log("Datos de Tren a Registrar:", formData);

    setTimeout(() => {
      setIsSubmitting(false);
      setMessage({
        type: "success",
        text: `Tren modelo "${formData.model}" registrado con éxito.`,
      });
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-neutral-900 p-8 flex justify-center">
      <div className="w-full max-w-2xl bg-neutral-800 p-8 rounded-xl border-2 border-yellow-500 shadow-xl shadow-yellow-500/10">
        <div className="flex items-center space-x-3 mb-6 border-b border-neutral-700 pb-4">
          <TrainIcon className="w-8 h-8 text-yellow-500" />
          <h2 className="text-2xl font-bold text-white">
            Registro de Tren
          </h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            <StyledInput label="Modelo" name="model" value={formData.model} onChange={handleChange} required />
            <StyledInput label="Fabricante" name="manufacturer" value={formData.manufacturer} onChange={handleChange} required />

            <StyledInput 
                label="Año de Fabricación" 
                name="yearBuilt" 
                value={formData.yearBuilt} 
                onChange={handleChange} 
                type="number" 
                required 
                min="1900" 
                max={new Date().getFullYear()}
            />
            
            <StyledSelect
                label="Tipo de Tren"
                name="trainType"
                value={formData.trainType}
                onChange={handleChange}
                required
                options={TRAIN_TYPES}
            />

            <StyledInput label="Peso (toneladas)" name="weight" value={formData.weight} onChange={handleChange} type="number" required step="0.1" min="0" />

            <div className="flex items-center space-x-2">
                <Gauge className="w-5 h-5 text-gray-400" />
                <StyledInput label="Velocidad Máxima (km/h)" name="maxSpeed" value={formData.maxSpeed} onChange={handleChange} type="number" required step="1" min="1" />
            </div>

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
            {isSubmitting ? "Guardando Tren..." : "Registrar Tren"}
          </button>
        </form>
      </div>
    </div>
  );
}