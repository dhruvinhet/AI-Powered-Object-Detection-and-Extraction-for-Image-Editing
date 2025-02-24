import React from "react";
import { Card, CardContent, Button } from "@mui/material";

const ExtractedImage = ({ extractedImage }) => {
  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = extractedImage;
    link.download = "extracted_object.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Card style={{ marginTop: 20, textAlign: "center" }}>
      <CardContent>
        <img src={extractedImage} alt="Extracted Object" style={{ width: "100%", maxHeight: 300, objectFit: "contain" }} />
        <Button variant="contained" onClick={handleDownload} style={{ marginTop: 10 }}>
          Download Extracted Object
        </Button>
      </CardContent>
    </Card>
  );
};

export default ExtractedImage;
