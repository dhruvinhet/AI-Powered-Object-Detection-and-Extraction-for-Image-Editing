import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { Container, Typography, Button, Slider, Box, Grid, Card, CardContent, CardMedia, Fade, Grow } from '@mui/material';
import { styled, keyframes } from '@mui/system';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import Lottie from 'lottie-react';
import loadingAnimation from './assets/loading.json';
import aiAnimation from './assets/ai-process.json'; // Add an AI-related animation JSON file
import { useDropzone } from 'react-dropzone';
import './App.css';
import ObjectDetection from './ObjectDetection';
import ExtractedImage from './ExtractedImage';

const pulse = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

const glow = keyframes`
  0% { box-shadow: 0 0 5px rgba(110, 133, 183, 0.5); }
  50% { box-shadow: 0 0 20px rgba(110, 133, 183, 0.8); }
  100% { box-shadow: 0 0 5px rgba(110, 133, 183, 0.5); }
`;

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#6e85b7" },
    secondary: { main: "#f4a261" },
    background: { default: "#1a1a2e" },
  },
  typography: {
    fontFamily: "'Inter', sans-serif",
    h3: { fontWeight: 700 },
    h6: { fontWeight: 500 },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          textTransform: 'none',
          padding: '12px 24px',
          boxShadow: '0 4px 14px rgba(0, 0, 0, 0.1)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 20px rgba(0, 0, 0, 0.2)',
            animation: `${pulse} 1.5s infinite`,
          }
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          background: 'rgba(255, 255, 255, 0.05)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          transition: 'all 0.4s ease',
          '&:hover': {
            transform: 'translateY(-6px)',
            boxShadow: '0 10px 30px rgba(0, 0, 0, 0.25)',
            animation: `${glow} 2s infinite`,
          }
        }
      }
    }
  }
});

const StyledSlider = styled(Slider)(({ theme }) => ({
  color: theme.palette.primary.main,
  '& .MuiSlider-thumb': {
    boxShadow: '0 0 0 8px rgba(110, 133, 183, 0.2)',
    transition: 'all 0.2s ease',
    '&:hover': {
      boxShadow: '0 0 0 12px rgba(110, 133, 183, 0.3)',
    }
  },
  '& .MuiSlider-rail': {
    opacity: 0.2,
    background: 'linear-gradient(90deg, #6e85b7, #f4a261)',
  }
}));

const DropZoneBox = styled(Box)(({ theme, isAnalyzing }) => ({
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: 16,
  padding: '40px',
  cursor: 'pointer',
  background: 'rgba(255, 255, 255, 0.03)',
  transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
  width: isAnalyzing ? '70vw' : '100%',
  maxWidth: isAnalyzing ? '900px' : '600px',
  '&:hover': {
    background: 'rgba(255, 255, 255, 0.06)',
    borderColor: theme.palette.secondary.main,
    transform: 'scale(1.02)',
  }
}));

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
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);

  const { getRootProps, getInputProps } = useDropzone({
    accept: "image/*",
    onDrop: (acceptedFiles) => {
      setSelectedFile(acceptedFiles[0]);
      setImage(acceptedFiles[0]);
    },
  });

  const handleUpload = async () => {
    if (!selectedFile) return alert("Please select an image first!");
    
    setLoading(true);
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/detect", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setTimeout(() => {
        setResult(response.data);
        setLoading(false);
        setIsAnalyzing(false);
      }, 3000);
    } catch (error) {
      console.error("Error uploading file:", error);
      setResult("Error detecting objects");
      setLoading(false);
      setIsAnalyzing(false);
    }
  };

  const handleImageUpload = (event) => {
    setSelectedFile(event.target.files[0]);
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
    setIsExtracting(true);
    const reader = new FileReader();
    reader.readAsDataURL(selectedFile);
    reader.onloadend = async () => {
      const base64Image = reader.result.split(',')[1];
      const response = await axios.post('http://localhost:5000/extract', {
        image: base64Image,
        box: detection.box,
        mask: detection.mask
      });

      if (response.data.image && response.data.image.trim() !== '') {
        setExtractedImage(`data:image/png;base64,${response.data.image}`);
        setOriginalExtractedImage(response.data.image);
      }
      setTimeout(() => setIsExtracting(false), 1500); // Animation duration
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
        setExtractedImage(`data:image/png;base64,${updatedImage}`);
      }
    };
  }, [brightness, contrast, saturation, blur, rotation, flipHorizontal, flipVertical]);

  const handleSave = () => {
    if (!extractedImage) return;
    const link = document.createElement('a');
    link.href = extractedImage;
    link.download = 'edited_image.png';
    link.click();
  };

 return (
  <ThemeProvider theme={theme}>
    <CssBaseline />
    <Container
    maxWidth="lg"
    sx={{
      minHeight: "100vh",
      background: 'linear-gradient(45deg, #1a1a2e 0%, #16213e 100%)',
      padding: '40px 20px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
    }}
    >
    <Fade in={true} timeout={1000}>
      <Box textAlign="center" mb={6}>
      <Typography 
        variant="h3" 
        sx={{ 
        background: 'linear-gradient(45deg, #6e85b7, #f4a261)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        textShadow: '0 2px 10px rgba(110, 133, 183, 0.5)',
        }}
      >
        AI Object Detection Studio
      </Typography>
      <Typography 
        variant="h6" 
        sx={{ 
        color: 'rgba(255, 255, 255, 0.7)',
        mt: 1,
        animation: `${pulse} 2s infinite`,
        }}
      >
        Powered by Advanced Computer Vision
      </Typography>
      </Box>
    </Fade>

    <DropZoneBox {...getRootProps()} isAnalyzing={isAnalyzing}>
      <input {...getInputProps()} />
      <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2, animation: `${glow} 3s infinite` }} />
      <Typography variant="h6" sx={{ color: 'white' }}>
      Drop your image here or click to upload
      </Typography>
      <Typography sx={{ color: 'rgba(255, 255, 255, 0.5)' }}>
      Supports all common image formats
      </Typography>
    </DropZoneBox>

    {selectedFile && (
      <Grow in={true} timeout={800}>
      <Card sx={{ mt: 4, width: '100%', maxWidth: isAnalyzing ? 800 : 600 }}>
        <CardMedia
        component="img"
        image={URL.createObjectURL(selectedFile)}
        alt="Uploaded Image"
        sx={{
          borderRadius: '12px',
          maxHeight: 400,
          objectFit: 'contain',
          background: 'rgba(255, 255, 255, 0.03)',
          filter: isAnalyzing ? 'brightness(1.1)' : 'none',
          transition: 'all 0.5s ease',
        }}
        />
      </Card>
      </Grow>
    )}

    <Button 
      variant="contained" 
      color="primary" 
      onClick={handleUpload} 
      disabled={!selectedFile || loading}
      sx={{ mt: 3, px: 6 }}
    >
      {loading ? (
      <Lottie animationData={loadingAnimation} style={{ width: 80, height: 80 }} />
      ) : (
      'Analyze Image'
      )}
    </Button>

    {result && (
      <Box mt={6} width="100%">
      <Typography variant="h5" sx={{ mb: 3, color: 'white', textShadow: '0 2px 8px rgba(0, 0, 0, 0.3)' }}>
        Detection Results
      </Typography>
      <Grid container spacing={3}>
        {result.map((obj, index) => (
        <Grid item xs={12} sm={6} md={4} key={index}>
          <Grow in={true} timeout={500 + index * 200}>
          <Card>
            <CardContent>
            <Typography variant="h6" sx={{ color: 'secondary.main' }}>
              {obj.label.toUpperCase()}
            </Typography>
            <Typography sx={{ color: 'rgba(255, 255, 255, 0.7)', mt: 1 }}>
              Confidence: {(obj.score * 100).toFixed(1)}%
            </Typography>
            <Box sx={{ mt: 2 }}>
              <StyledSlider
              value={obj.score * 100}
              max={100}
              disabled
              />
            </Box>
            <Button
              variant="outlined"
              color="secondary"
              onClick={() => handleExtract(obj)}
              sx={{ mt: 2, width: '100%' }}
            >
              Extract Object
            </Button>
            {isExtracting && (
              <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <Lottie animationData={aiAnimation} style={{ width: 150, height: 150 }} />
              </Box>
            )}
            </CardContent>
          </Card>
          </Grow>
        </Grid>
        ))}
      </Grid>
      </Box>
    )}

    {extractedImage && (
      <Box mt={6} width="100%" display="flex" justifyContent="center">
      <Card sx={{ p: 3, position: 'relative', maxWidth: 600 }}>
        <Typography variant="h5" sx={{ mb: 3, color: 'white', textShadow: '0 2px 8px rgba(0, 0, 0, 0.3)' }}>
        Image Editor
        </Typography>
        <Box sx={{ position: 'relative', mb: 3 }}>
        <img 
          src={extractedImage} 
          alt="Extracted" 
          style={{ 
          maxWidth: '100%', 
          borderRadius: 8,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)',
          transition: 'all 0.3s ease',
          }} 
        />
        <Box
          sx={{
          position: 'absolute',
          top: -10,
          left: -10,
          right: -10,
          bottom: -10,
          borderRadius: 12,
          background: 'radial-gradient(circle, rgba(110, 133, 183, 0.2) 0%, rgba(110, 133, 183, 0) 70%)',
          animation: `${pulse} 2s infinite`,
          zIndex: -1,
          }}
        />
        </Box>
        
        {/* Editor Controls */}
        {[
        { label: 'Brightness', value: brightness, min: 0.5, max: 2, step: 0.1, set: setBrightness },
        { label: 'Contrast', value: contrast, min: 0.5, max: 2, step: 0.1, set: setContrast },
        { label: 'Saturation', value: saturation, min: 0.5, max: 2, step: 0.1, set: setSaturation },
        { label: 'Blur', value: blur, min: 0, max: 10, step: 0.5, set: setBlur },
        { label: 'Rotate', value: rotation, min: -180, max: 180, step: 1, set: setRotation },
        ].map((control) => (
        <Box key={control.label} sx={{ mb: 2 }}>
          <Typography sx={{ color: 'white', mb: 1 }}>{control.label}</Typography>
          <StyledSlider
          value={control.value}
          min={control.min}
          max={control.max}
          step={control.step}
          onChange={(e, val) => control.set(val)}
          />
        </Box>
        ))}

        <Grid container spacing={2} sx={{ mt: 2 }}>
        <Grid item xs={6}>
          <Button
          variant="outlined"
          color="secondary"
          onClick={() => setFlipHorizontal(!flipHorizontal)}
          fullWidth
          >
          Flip H
          </Button>
        </Grid>
        <Grid item xs={6}>
          <Button
          variant="outlined"
          color="secondary"
          onClick={() => setFlipVertical(!flipVertical)}
          fullWidth
          >
          Flip V
          </Button>
        </Grid>
        </Grid>

        <Button
        variant="contained"
        color="primary"
        onClick={handleSave}
        sx={{ mt: 3, width: '100%' }}
        >
        Save Edited Image
        </Button>
      </Card>
      </Box>
    )}
    </Container>
  </ThemeProvider>
  );
}

export default App;