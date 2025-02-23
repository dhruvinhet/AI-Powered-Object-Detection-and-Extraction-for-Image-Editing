import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Typography, Button, Slider, Box, Grid, Card, CardContent, CardMedia } from '@mui/material';
import { styled } from '@mui/system';
import './App.css';

const Input = styled('input')({
  display: 'none',
});

function App() {
  const [image, setImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [extractedImage, setExtractedImage] = useState(null);
  const [originalExtractedImage, setOriginalExtractedImage] = useState(null);
  const [brightness, setBrightness] = useState(1);
  const [contrast, setContrast] = useState(1);
  const [saturation, setSaturation] = useState(1);
  const [blur, setBlur] = useState(0);
  const [rotation, setRotation] = useState(0);
  const [flipHorizontal, setFlipHorizontal] = useState(false);
  const [flipVertical, setFlipVertical] = useState(false);

  const handleImageUpload = (event) => {
    setImage(event.target.files[0]);
  };

  const handleDetect = async () => {
    if (!image) return;
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await axios.post('http://localhost:5000/detect', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setDetections(response.data);
    } catch (error) {
      console.error("Error detecting objects:", error);
    }
  };

  const handleExtract = async (detection) => {
    const reader = new FileReader();
    reader.readAsDataURL(image);
    reader.onloadend = async () => {
      const base64Image = reader.result.split(',')[1];
      const response = await axios.post('http://localhost:5000/extract', {
        image: base64Image,
        box: detection.box,
        mask: detection.mask
      });

      // Check for valid image before setting state
      if (response.data.image && response.data.image.trim() !== '') {
        console.log("Extracted Image:", response.data.image);  // Debugging
        setExtractedImage(response.data.image);
        setOriginalExtractedImage(response.data.image);
      }
    };
  };

  useEffect(() => {
    if (!originalExtractedImage) return;

    const img = new Image();
    img.src = `data:image/png;base64,${originalExtractedImage}`;
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.filter = `brightness(${brightness}) contrast(${contrast}) saturate(${saturation}) blur(${blur}px)`;
      ctx.setTransform(1, 0, 0, 1, 0, 0);

      ctx.translate(canvas.width / 2, canvas.height / 2);
      if (flipHorizontal) ctx.scale(-1, 1);
      if (flipVertical) ctx.scale(1, -1);
      ctx.rotate((rotation * Math.PI) / 180);
      ctx.translate(-canvas.width / 2, -canvas.height / 2);

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      const updatedImage = canvas.toDataURL('image/png').split(',')[1];

      if (updatedImage && updatedImage.trim() !== '') {
        setExtractedImage(updatedImage);
      }
    };
  }, [brightness, contrast, saturation, blur, rotation, flipHorizontal, flipVertical]);

  const handleSave = () => {
    if (!extractedImage) return;
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${extractedImage}`;
    link.download = 'edited_image.png';
    link.click();
  };

  return (
    <Container>
      <Typography variant="h3" gutterBottom>Image Editor</Typography>
      <Box mb={2}>
        <label htmlFor="upload-button">
          <Input accept="image/*" id="upload-button" type="file" onChange={handleImageUpload} />
          <Button variant="contained" component="span" sx={{ marginRight: 2 }}>Upload Image</Button>
        </label>
        <Button variant="contained" onClick={handleDetect} disabled={!image}>Detect Objects</Button>
      </Box>

      {detections.length > 0 && (
        <Box mt={2}>
          <Typography variant="h5">Detected Objects</Typography>
          <Grid container spacing={2}>
            {detections.map((detection, index) => (
              <Grid item key={index}>
                <Card>
                  <CardContent>
                    <Typography variant="h6">{detection.label}</Typography>
                    <Button variant="contained" onClick={() => handleExtract(detection)}>Extract</Button>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {extractedImage && extractedImage.trim() !== '' && (
        <Box mt={4}>
          <Typography variant="h5" gutterBottom>Extracted Image</Typography>
          <Card>
            
          <img 
            src={`data:image/png;base64,${extractedImage}`} 
            alt="Extracted" 
            style={{ maxWidth: '100%', display: 'block', margin: 'auto', backgroundColor: 'transparent' }} 
          />
          </Card>
          <Box>
            <Typography>Brightness</Typography>
            <Slider value={brightness} min={0.5} max={2} step={0.1} onChange={(e, val) => setBrightness(val)} />
            <Typography>Contrast</Typography>
            <Slider value={contrast} min={0.5} max={2} step={0.1} onChange={(e, val) => setContrast(val)} />
            <Typography>Saturation</Typography>
            <Slider value={saturation} min={0.5} max={2} step={0.1} onChange={(e, val) => setSaturation(val)} />
            <Typography>Blur</Typography>
            <Slider value={blur} min={0} max={10} step={0.5} onChange={(e, val) => setBlur(val)} />
            <Typography>Rotate</Typography>
            <Slider value={rotation} min={-180} max={180} step={1} onChange={(e, val) => setRotation(val)} />

            <Box mt={2}>
              <Button variant="contained" onClick={() => setFlipHorizontal(!flipHorizontal)} sx={{ marginRight: 2 }}>
                Flip Horizontal
              </Button>
              <Button variant="contained" onClick={() => setFlipVertical(!flipVertical)}>
                Flip Vertical
              </Button>
            </Box>

            <Box mt={3}>
              <Button variant="contained" onClick={handleSave} sx={{ width: '100%', padding: 1.5 }}>
                Save Image
              </Button>
            </Box>
          </Box>
        </Box>
      )}
    </Container>
  );
}

export default App; 
