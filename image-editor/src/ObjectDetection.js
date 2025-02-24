import React from "react";
import { Card, CardContent, Button } from "@mui/material";
import axios from "axios";

const ObjectDetection = ({ image, detections, setSelectedObject, setExtractedImage }) => {
  const handleExtract = async (detection) => {
    setSelectedObject(detection);

    const payload = {
      image: image.split(",")[1], // Base64 encoding
      box: detection.box,
      mask: detection.mask,
    };

    try {
      const response = await axios.post("http://127.0.0.1:5000/extract", payload);
      setExtractedImage(`data:image/png;base64,${response.data.image}`);
    } catch (error) {
      console.error("Error extracting object:", error);
    }
  };

  return (
    <Card style={{ marginTop: 20, textAlign: "center" }}>
      <CardContent>
        <img src={image} alt="Uploaded" style={{ width: "100%", maxHeight: 300, objectFit: "contain" }} />
        <div>
          {detections.map((det, index) => (
            <Button key={index} variant="contained" style={{ margin: 5 }} onClick={() => handleExtract(det)}>
              {det.label} ({(det.score * 100).toFixed(1)}%)
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default ObjectDetection;
