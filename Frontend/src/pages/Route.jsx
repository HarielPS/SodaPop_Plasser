import { useState } from "react";
import { ChevronRight, Map, TrainFront } from "lucide-react";

// Mockup de Estaciones para Selects
const mockStations = [
  { id: 1, name: "Estación Central", city: "CDMX" },
  { id: 2, name: "Terminal Pacífico", city: "Guadalajara" },
  { id: 3, name: "Puerto Veracruz", city: "Veracruz" },
];

const StyledInput = ({ label, name, value, onChange, type = "text", required = false, step }) => (
    <div>
        <label className="block text-sm font-medium text-gray-300">{label}</label>
        <input
            type={type}
            name={name}
            value={value}
            onChange={onChange}
            required={required}
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
            <option value="" disabled>Seleccione una opción</option>
            {options.map((option) => (
                <option key={option.id} value={option.id} className="bg-neutral-800 text-white">
                    {option.name || option.description} ({option.city})
                </option>
            ))}
        </select>
    </div>
);

export default function RouteRegistration() {
  const [formData, setFormData] = useState({
    routeCode: "",
    originId: "",
    destinationId: "",
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

    if (formData.originId === formData.destinationId) {
        setIsSubmitting(false);
        setMessage({
            type: "error",
            text: "El origen y el destino de la ruta no pueden ser la misma estación.",
        });
        return;
    }

    console.log("Datos de Ruta a Registrar:", formData);

    setTimeout(() => {
      setIsSubmitting(false);
      setMessage({
        type: "success",
        text: `Ruta "${formData.routeCode}" registrada exitosamente.`,
      });
    }, 1500);
  };

  const isFormInvalid = !formData.routeCode || !formData.originId || !formData.destinationId || (formData.originId === formData.destinationId);


  return (
    <div className="min-h-screen bg-neutral-900 p-8 flex justify-center">
      <div className="w-full max-w-xl bg-neutral-800 p-8 rounded-xl border-2 border-yellow-500 shadow-xl shadow-yellow-500/10">
        <div className="flex items-center space-x-3 mb-6 border-b border-neutral-700 pb-4">
          <Map className="w-8 h-8 text-yellow-500" />
          <h2 className="text-2xl font-bold text-white">
            Registro de Ruta Ferroviaria
          </h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <StyledInput label="Código de Ruta (único)" name="routeCode" value={formData.routeCode} onChange={handleChange} required />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
            <StyledSelect 
                label="Estación de Origen" 
                name="originId" 
                value={formData.originId} 
                onChange={handleChange} 
                required 
                options={mockStations} 
            />
            
            <StyledSelect 
                label="Estación de Destino" 
                name="destinationId" 
                value={formData.destinationId} 
                onChange={handleChange} 
                required 
                options={mockStations} 
            />
          </div>
          <div className="flex justify-center text-yellow-500 pt-2 pb-4">
                <TrainFront className="w-6 h-6 mr-2" />
                <ChevronRight className="w-6 h-6" />
                <ChevronRight className="w-6 h-6 -ml-3" />
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
            {isSubmitting ? "Guardando Ruta..." : "Registrar Ruta"}
          </button>
        </form>
      </div>
    </div>
  );
}