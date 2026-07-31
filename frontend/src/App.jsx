import { useState } from "react";
import { verifyDocument } from "./api";
import UploadForm from "./components/UploadForm";
import VerificationReport from "./components/VerificationReport";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(file) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await verifyDocument(file);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <h1>AI Identity Shield</h1>
      <p className="tagline">Upload a document image to run forgery detection and OCR field extraction.</p>

      <UploadForm onSubmit={handleSubmit} loading={loading} />

      {error && <p className="error-message">{error}</p>}
      {result && <VerificationReport result={result} />}
    </main>
  );
}
