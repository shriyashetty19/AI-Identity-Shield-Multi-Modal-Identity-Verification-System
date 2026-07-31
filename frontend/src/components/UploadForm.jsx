import { useState } from "react";

export default function UploadForm({ onSubmit, loading }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  function handleFileChange(e) {
    const selected = e.target.files?.[0] ?? null;
    setFile(selected);
    setPreviewUrl(selected ? URL.createObjectURL(selected) : null);
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (file) onSubmit(file);
  }

  return (
    <form className="upload-form" onSubmit={handleSubmit}>
      <label className="upload-dropzone" htmlFor="document-input">
        {previewUrl ? (
          <img src={previewUrl} alt="Selected document preview" className="upload-preview" />
        ) : (
          <span>Click to choose a document image (passport, ID card, driver's license)</span>
        )}
      </label>
      <input
        id="document-input"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        hidden
      />
      <button type="submit" disabled={!file || loading}>
        {loading ? "Verifying..." : "Run verification"}
      </button>
    </form>
  );
}
