import { ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";

export function BackButton({ fallback = "/app/browse" }: { fallback?: string }) {
  const navigate = useNavigate();

  function goBack() {
    navigate(fallback);
  }

  return (
    <button type="button" className="back-button" onClick={goBack}>
      <ArrowLeft size={17} />
      <span>Back</span>
    </button>
  );
}
