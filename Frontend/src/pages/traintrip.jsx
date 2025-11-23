import { useState } from "react";
import { Calendar, Globe, Clock, Weight } from "lucide-react";

// Mockup de Datos Relacionados
const mockTrains = [
  { id: 101, model: "T-2000", type: "Cargo" },
  { id: 102, model: "Hyperion-V", type: "High-Speed" },
];

const mockRoutes = [
  { id: 1, code: "R-CDMX-GDL", description: "CDMX -> Guadalajara" },
  { id: 2, code: "R-VER-MEX", description: "Veracruz -> CDMX" },
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

const StyledSelect = ({ label, name, value, onChange, required, options, getLabel }) => (
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
                <option key={option.id} value={option.id} className="bg-neutral-800 text-white">
                    {getLabel(option)}
                </option>
            ))}
        </select>
    </div>
);


export default function TrainTripRegistration() {
  const [formData, setFormData] = useState({
    trainId: "",
    routeId: "",
    departureTime: "",
    arrivalTime: "",
    cargoWeight: "",
    averageSpeed: "",
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    const newValue = (name === 'cargoWeight' || name === 'averageSpeed') ? parseFloat(value) : value;

    setFormData((prevData) => ({
      ...prevData,
      [name]: newValue,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setMessage(null);

    if (new Date(formData.departureTime) >= new Date(formData.arrivalTime)) {
        setIsSubmitting(false);
        setMessage({
            type: "error",
            text: "La hora de partida debe ser anterior a la hora de llegada.",
        });
        return;
    }

    console.log("Datos de Viaje de Tren a Registrar:", formData);

    setTimeout(() => {
      setIsSubmitting(false);
      setMessage({
        type: "success",
        text: `Viaje de Tren (Tren ID: ${formData.trainId}, Ruta: ${formData.routeId}) registrado.`,
      });
    }, 1500);
  };

  const isFormInvalid = !formData.trainId || !formData.routeId || !formData.departureTime || !formData.arrivalTime;

  return (
    <div className="min-h-screen bg-neutral-900 p-8 flex justify-center">
      <div className="w-full max-w-2xl bg-neutral-800 p-8 rounded-xl border-2 border-yellow-500 shadow-xl shadow-yellow-500/10">
        <div className="flex items-center space-x-3 mb-6 border-b border-neutral-700 pb-4">
          <Globe className="w-8 h-8 text-yellow-500" />
          <h2 className="text-2xl font-bold text-white">
            Registro de Viaje de Tren (Train Trip)
          </h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            <StyledSelect
                label="Seleccionar Tren"
                name="trainId"
                value={formData.trainId}
                onChange={handleChange}
                required
                options={mockTrains}
                getLabel={(t) => `${t.id} - ${t.model} (${t.type})`}
            />

            <StyledSelect
                label="Seleccionar Ruta"
                name="routeId"
                value={formData.routeId}
                onChange={handleChange}
                required
                options={mockRoutes}
                getLabel={(r) => `${r.code} - ${r.description}`}
            />
            
            <div className="flex items-center space-x-2">
                <Calendar className="w-5 h-5 text-gray-400" />
                <StyledInput label="Hora de Partida" name="departureTime" value={formData.departureTime} onChange={handleChange} type="datetime-local" required />
            </div>

            <div className="flex items-center space-x-2">
                <Clock className="w-5 h-5 text-gray-400" />
                <StyledInput label="Hora de Llegada" name="arrivalTime" value={formData.arrivalTime} onChange={handleChange} type="datetime-local" required />
            </div>
            
            <div className="flex items-center space-x-2">
                <Weight className="w-5 h-5 text-gray-400" />
                <StyledInput label="Peso de Carga (toneladas)" name="cargoWeight" value={formData.cargoWeight} onChange={handleChange} type="number" required step="0.01" min="0" />
            </div>

            <div className="flex items-center space-x-2">
                <Globe className="w-5 h-5 text-gray-400" />
                <StyledInput label="Velocidad Promedio (km/h)" name="averageSpeed" value={formData.averageSpeed} onChange={handleChange} type="number" required step="0.1" min="1" />
            </div>
          </div>

          {message && (
            <div className={`p-3 rounded-lg text-sm ${message.type === 'success' ? 'bg-green-700 text-white' : 'bg-red-700 text-white'}`}>
              {message.text}
            </div>
          )}

          <button
            type="submit"
            disabled={isSubmitting || isFormInvalid}
            className={`w-full py-3 px-4 border border-transparent rounded-lg shadow-lg text-lg font-bold text-black transition duration-150 ${
              isSubmitting || isFormInvalid
                ? 'bg-yellow-400 cursor-not-allowed' 
                : 'bg-yellow-500 hover:bg-yellow-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500'
            }`}
          >
            {isSubmitting ? "Programando Viaje..." : "Registrar Viaje"}
          </button>
        </form>
      </div>
    </div>
  );
}