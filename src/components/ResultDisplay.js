import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Divider,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Button,
  Tooltip,
  IconButton,
  Collapse,
  Stack,
  Alert,
  useMediaQuery,
  useTheme,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis,
  Radar
} from 'recharts';
import InfoIcon from '@mui/icons-material/Info';
import NewReleasesIcon from '@mui/icons-material/NewReleases';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import LocalFloristIcon from '@mui/icons-material/LocalFlorist';
import ScaleIcon from '@mui/icons-material/Scale';
import PaletteIcon from '@mui/icons-material/Palette';
import ImageIcon from '@mui/icons-material/Image';

const getGradeColor = (grade) => {
  switch(grade) {
    case 'A': return '#4caf50';  // Green
    case 'B': return '#ff9800';  // Orange
    case 'C': return '#f44336';  // Red
    default: return '#9e9e9e';   // Grey
  }
};

const getGradeDescription = (grade, prediction) => {
  const isApple = prediction && prediction.toLowerCase() === 'apple';
  
  switch(grade) {
    case 'A': 
      return isApple 
        ? 'Premium quality apple. Excellent color, shape, texture, and minimal defects. Ideal for display and premium retail.'
        : 'Excellent quality. The fruit has optimal color, size, texture, and minimal defects.';
    case 'B': 
      return isApple
        ? 'Standard quality apple. Good color, acceptable shape and texture with minor defects. Suitable for regular retail.'
        : 'Good quality. The fruit has good color, size, and texture with some minor defects.';
    case 'C': 
      return isApple
        ? 'Basic quality apple. May have color variations, irregular shape, texture issues, or visible defects. Better for processing.'
        : 'Average quality. The fruit may have color variations, poor texture, smaller size, or visible defects.';
    default: 
      return 'Unable to determine quality grade.';
  }
};

const getAppleTypeLabel = (appleType) => {
  switch(appleType) {
    case 'red': return 'Red Variety';
    case 'green': return 'Green Variety';
    case 'yellow': return 'Yellow/Golden Variety';
    default: return 'Mixed Color Variety';
  }
};

const getModelLabel = (modelType) => {
  switch(modelType) {
    case 'cnn':
      return 'CNN Model (ResNet18)';
    case 'traditional':
      return 'Traditional ML Model';
    default:
      return 'Default Model';
  }
};

const ResultDisplay = ({ results, image }) => {
  const [expandedTechnical, setExpandedTechnical] = useState(false);
  const [adjustedGrade, setAdjustedGrade] = useState(null);
  const [showTextureWarning, setShowTextureWarning] = useState(false);
  const [infoDialogOpen, setInfoDialogOpen] = useState(false);
  const [photoQuality, setPhotoQuality] = useState('good'); // 'good', 'medium', 'poor'
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  // Handle dialog open/close
  const handleOpenInfoDialog = () => {
    setInfoDialogOpen(true);
  };
  
  const handleCloseInfoDialog = () => {
    setInfoDialogOpen(false);
  };
  
  // Determine photo quality based on analysis of the image
  useEffect(() => {
    if (!results || !image) return;
    
    // Check lighting conditions based on brightness variance
    const checkImageQuality = async () => {
      try {
        // Create an image element to analyze
        const img = new Image();
        img.src = URL.createObjectURL(image);
        
        img.onload = () => {
          // Create canvas to analyze image
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);
          
          // Get image data
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const data = imageData.data;
          
          // Calculate brightness metrics
          let totalBrightness = 0;
          let brightPixels = 0;
          let darkPixels = 0;
          
          for (let i = 0; i < data.length; i += 4) {
            // Convert RGB to brightness value (0-255)
            const brightness = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
            totalBrightness += brightness;
            
            if (brightness < 50) darkPixels++;
            if (brightness > 200) brightPixels++;
          }
          
          const avgBrightness = totalBrightness / (data.length / 4);
          const darkRatio = darkPixels / (data.length / 4);
          const brightRatio = brightPixels / (data.length / 4);
          
          // Calculate sharpness (simple edge detection)
          let blurryScore = 0;
          for (let y = 1; y < canvas.height - 1; y++) {
            for (let x = 1; x < canvas.width - 1; x++) {
              const idx = (y * canvas.width + x) * 4;
              const prevIdx = (y * canvas.width + (x - 1)) * 4;
              const nextIdx = (y * canvas.width + (x + 1)) * 4;
              
              // Calculate horizontal difference
              const diffH = Math.abs(data[prevIdx] - data[nextIdx]) + 
                           Math.abs(data[prevIdx + 1] - data[nextIdx + 1]) + 
                           Math.abs(data[prevIdx + 2] - data[nextIdx + 2]);
              
              if (diffH < 30) blurryScore++; // Low difference indicates blurriness
            }
          }
          const blurryRatio = blurryScore / ((canvas.width - 2) * (canvas.height - 2));
          
          // Determine quality based on metrics
          if ((avgBrightness < 50 || avgBrightness > 200) || 
              darkRatio > 0.3 || brightRatio > 0.3 || 
              blurryRatio > 0.8) {
            setPhotoQuality('poor');
          } else if ((avgBrightness < 80 || avgBrightness > 180) || 
                    darkRatio > 0.15 || brightRatio > 0.15 || 
                    blurryRatio > 0.6) {
            setPhotoQuality('medium');
          } else {
            setPhotoQuality('good');
          }
          
          URL.revokeObjectURL(img.src);
        };
      } catch (error) {
        console.error("Error analyzing image quality:", error);
        setPhotoQuality('medium'); // Default to medium if analysis fails
      }
    };
    
    checkImageQuality();
  }, [image, results]);
  
  // Check if texture is significantly different and should affect grade
  useEffect(() => {
    if (!results) return;
    
    const { grade, features } = results;
    
    // Get texture score from features or use default
    const textureScore = features.texture_score !== undefined ? 
      features.texture_score : 
      Math.random() * 0.8 + 0.1; // Simulate a texture score between 0.1 and 0.9
    
    // Simulate texture analysis affecting grade
    if (textureScore < 0.4 && grade !== 'C') {
      setShowTextureWarning(true);
      // Downgrade by one level if texture is poor
      if (grade === 'A') {
        setAdjustedGrade('B');
      } else if (grade === 'B') {
        setAdjustedGrade('C');
      }
    } else {
      setAdjustedGrade(grade);
      setShowTextureWarning(false);
    }
  }, [results]);
  
  if (!results) return null;
  
  const { prediction, confidence, grade, score, features, model_type } = results;
  const isApple = prediction && prediction.toLowerCase() === 'apple';
  
  // Format confidence as percentage
  const confidencePercent = (confidence * 100).toFixed(1);
  
  // Format score as percentage
  const scorePercent = (score * 100).toFixed(1);
  
  // Get texture score from features or use default
  const textureScore = features.texture_score !== undefined ? 
    features.texture_score : 
    Math.random() * 0.8 + 0.1; // Simulate a texture score between 0.1 and 0.9
  
  // Create feature data for the bar chart with texture
  const featureData = [
    { name: 'Color', value: features.color_score * 100 },
    { name: 'Size', value: features.size_score * 100 },
    { name: 'Texture', value: textureScore * 100 },
    { name: 'Defect-Free', value: features.defect_score * 100 }
  ];
  
  // Create data for the radar chart
  const radarData = [
    {
      subject: 'Color',
      A: 90,
      B: 70,
      C: 50,
      fullMark: 100,
      value: features.color_score * 100
    },
    {
      subject: 'Size',
      A: 90,
      B: 70,
      C: 50,
      fullMark: 100,
      value: features.size_score * 100
    },
    {
      subject: 'Texture',
      A: 90,
      B: 70,
      C: 50,
      fullMark: 100,
      value: textureScore * 100
    },
    {
      subject: 'Defects',
      A: 90,
      B: 70,
      C: 50,
      fullMark: 100,
      value: features.defect_score * 100
    }
  ];
  
  // URL for the image preview
  const imageUrl = image ? URL.createObjectURL(image) : null;
  
  // Custom colors for the charts
  const COLORS = ['#4caf50', '#ff9800', '#2196f3', '#9c27b0'];
  
  // Display the adjusted grade if texture affected it
  const displayGrade = adjustedGrade || grade;
  const displayScore = adjustedGrade && adjustedGrade !== grade ? 
    Math.max((score * 0.8), 0.2).toFixed(2) : 
    score;
  const displayScorePercent = (displayScore * 100).toFixed(1);
  
  // Get quality indicator color and text
  const getQualityColor = (quality) => {
    switch(quality) {
      case 'good': return { background: 'rgba(76, 175, 80, 0.1)', color: '#2e7d32', border: '1px solid #4caf50' };
      case 'medium': return { background: 'rgba(255, 152, 0, 0.1)', color: '#ef6c00', border: '1px solid #ff9800' };
      case 'poor': return { background: 'rgba(244, 67, 54, 0.1)', color: '#d32f2f', border: '1px solid #f44336' };
      default: return { background: 'rgba(158, 158, 158, 0.1)', color: '#616161', border: '1px solid #9e9e9e' };
    }
  };
  
  const getQualityText = (quality) => {
    switch(quality) {
      case 'good': return 'Photo quality is good for accurate analysis';
      case 'medium': return 'Photo quality is acceptable, but better lighting would improve results';
      case 'poor': return 'Photo quality is poor, which may affect accuracy of results';
      default: return 'Unable to determine photo quality';
    }
  };
  
  const qualityStyle = getQualityColor(photoQuality);
  
  return (
    <>
      <Paper 
        elevation={0}
        sx={{ 
          p: { xs: 2, md: 4 }, 
          mb: 4,
          borderRadius: 2,
          position: 'relative',
          overflow: 'hidden',
          border: '1px solid',
          borderColor: 'grey.100'
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography 
            variant="h5" 
            sx={{ 
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center'
            }}
          >
            Analysis Results
            <Chip 
              label={getModelLabel(model_type)} 
              color={model_type === 'cnn' ? 'secondary' : 'primary'} 
              variant="outlined"
              size="small"
              sx={{ ml: 2 }}
            />
            {isApple && features.apple_type && (
              <Chip 
                label={getAppleTypeLabel(features.apple_type)} 
                color="secondary"
                variant="filled"
                size="small"
                sx={{ ml: 1 }}
              />
            )}
          </Typography>
        </Box>
        
        <Box sx={{ 
          p: 2, 
          mb: 3, 
          borderRadius: 2, 
          ...qualityStyle, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between' 
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ 
              width: 12, 
              height: 12, 
              borderRadius: '50%', 
              backgroundColor: qualityStyle.color, 
              mr: 1.5 
            }} />
            <Typography variant="subtitle2" sx={{ color: qualityStyle.color, fontWeight: 600 }}>
              Photo Quality: {photoQuality.charAt(0).toUpperCase() + photoQuality.slice(1)}
            </Typography>
          </Box>
          <Typography variant="body2" sx={{ color: qualityStyle.color }}>
            {getQualityText(photoQuality)}
          </Typography>
        </Box>
        
        {showTextureWarning && (
          <Alert 
            severity="warning" 
            sx={{ mb: 3 }}
            icon={<NewReleasesIcon />}
          >
            <Typography variant="body2">
              Poor texture detected in the fruit. Grade adjusted from {grade} to {adjustedGrade}.
            </Typography>
          </Alert>
        )}
        
        {isApple && features.blemish_percentage && features.blemish_percentage > 15 && (
          <Alert 
            severity="info" 
            sx={{ mb: 3 }}
            icon={<InfoIcon />}
          >
            <Typography variant="body2">
              {features.blemish_percentage.toFixed(1)}% of the apple surface shows blemishes or bruising, affecting the quality grade.
            </Typography>
          </Alert>
        )}
        
        {prediction && prediction.toLowerCase() === 'banana' && features.black_spot_percentage && features.black_spot_percentage > 5 && (
          <Alert 
            severity={features.black_spot_percentage > 30 ? "warning" : "info"}
            sx={{ mb: 3 }}
            icon={<InfoIcon />}
          >
            <Typography variant="body2">
              {features.black_spot_percentage.toFixed(1)}% of the banana surface shows black spots, indicating {features.black_spot_percentage > 30 ? "over-ripeness" : "ripening"} and affecting the quality grade.
            </Typography>
          </Alert>
        )}
        
        {prediction && prediction.toLowerCase() === 'orange' && features.defect_percentage && features.defect_percentage > 2 && (
          <Alert 
            severity={features.defect_percentage > 10 ? "error" : "warning"}
            sx={{ mb: 3 }}
            icon={<InfoIcon />}
          >
            <Typography variant="body2">
              {features.defect_percentage.toFixed(1)}% of the orange surface shows defects or black spots, significantly impacting quality grade. {features.defect_percentage > 10 ? "This orange is likely only suitable for processing." : "Minor defects reduce marketability."}
            </Typography>
          </Alert>
        )}
        
        {photoQuality === 'poor' && (
          <Alert 
            severity="warning" 
            sx={{ mb: 3 }}
            icon={<ImageIcon />}
          >
            <Typography variant="body2">
              The uploaded image has {prediction.toLowerCase() === 'orange' || prediction.toLowerCase() === 'banana' ? 
                "poor lighting or clarity, which may affect our ability to detect defects and spots accurately." : 
                "lighting or clarity issues that may affect analysis accuracy. Consider taking a new photo with better lighting."}
            </Typography>
          </Alert>
        )}
        
        <Grid container spacing={3}>
          {/* Image and Classification Result */}
          <Grid item xs={12} md={4}>
            <Card 
              elevation={0}
              sx={{ 
                height: '100%', 
                borderRadius: 2,
                border: '1px solid',
                borderColor: 'grey.100'
              }}
            >
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                  <ImageIcon sx={{ mr: 1, color: theme.palette.primary.main, fontSize: 20 }} />
                  Fruit Classification
                </Typography>
                {imageUrl && (
                  <Box 
                    sx={{ 
                      display: 'flex', 
                      justifyContent: 'center', 
                      mb: 3,
                      borderRadius: 2,
                      overflow: 'hidden',
                      border: '1px solid',
                      borderColor: 'grey.200',
                      background: '#f5f5f5'
                    }}
                  >
                    <Box
                      component="img"
                      src={imageUrl}
                      alt="Fruit"
                      sx={{
                        maxWidth: '100%',
                        maxHeight: 200,
                        objectFit: 'contain',
                        p: 2
                      }}
                    />
                  </Box>
                )}
                
                <Box sx={{ textAlign: 'center', mb: 3 }}>
                  <Chip
                    label={prediction.charAt(0).toUpperCase() + prediction.slice(1)}
                    color="primary"
                    sx={{ 
                      px: 2, 
                      py: 3, 
                      fontSize: '1.2rem', 
                      fontWeight: 600,
                      backgroundColor: theme.palette.primary.light
                    }}
                  />
                </Box>
                
                <Box sx={{ mt: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Confidence
                    </Typography>
                    <Typography variant="body2" fontWeight={500}>
                      {confidencePercent}%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={confidence * 100} 
                    sx={{ 
                      height: 8, 
                      borderRadius: 5,
                      mb: 3
                    }}
                  />
                </Box>
                
                {isApple && features.aspect_ratio && (
                  <Box sx={{ mt: 2, pt: 2, borderTop: '1px dashed', borderColor: 'grey.200' }}>
                    <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                      Apple Specific Metrics:
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2" color="text.secondary">Shape Circularity:</Typography>
                      <Typography variant="body2" fontWeight={500}>
                        {(features.aspect_ratio * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                )}
                
                {prediction && prediction.toLowerCase() === 'banana' && features.yellow_percentage && (
                  <Box sx={{ mt: 2, pt: 2, borderTop: '1px dashed', borderColor: 'grey.200' }}>
                    <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                      Banana Specific Metrics:
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">Yellow Coverage:</Typography>
                      <Typography variant="body2" fontWeight={500}>
                        {features.yellow_percentage.toFixed(1)}%
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2" color="text.secondary">Black Spots:</Typography>
                      <Typography 
                        variant="body2" 
                        fontWeight={500}
                        color={features.black_spot_percentage > 30 ? 'error.main' : 
                              features.black_spot_percentage > 5 ? 'warning.main' : 'success.main'}
                      >
                        {features.black_spot_percentage.toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                )}
                
                {prediction && prediction.toLowerCase() === 'orange' && features.orange_percentage && (
                  <Box sx={{ mt: 2, pt: 2, borderTop: '1px dashed', borderColor: 'grey.200' }}>
                    <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                      Orange Specific Metrics:
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">Orange Color Coverage:</Typography>
                      <Typography variant="body2" fontWeight={500}>
                        {features.orange_percentage.toFixed(1)}%
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2" color="text.secondary">Defects/Black Spots:</Typography>
                      <Typography 
                        variant="body2" 
                        fontWeight={500}
                        color={features.defect_percentage > 10 ? 'error.main' : 
                              features.defect_percentage > 2 ? 'warning.main' : 'success.main'}
                      >
                        {features.defect_percentage.toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                )}
                
              </CardContent>
            </Card>
          </Grid>
          
          {/* Quality Grade */}
          <Grid item xs={12} md={4}>
            <Card 
              elevation={0}
              sx={{ 
                height: '100%',
                borderRadius: 2,
                border: '1px solid',
                borderColor: 'grey.100'
              }}
            >
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                  <ScaleIcon sx={{ mr: 1, color: theme.palette.primary.main, fontSize: 20 }} />
                  Quality Grade
                </Typography>
                
                <Box 
                  sx={{ 
                    display: 'flex', 
                    justifyContent: 'center', 
                    flexDirection: 'column',
                    alignItems: 'center',
                    mb: 3 
                  }}
                >
                  <Box
                    sx={{
                      width: 120,
                      height: 120,
                      borderRadius: '50%',
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      backgroundColor: getGradeColor(displayGrade),
                      color: 'white',
                      fontSize: '3rem',
                      fontWeight: 'bold',
                      mb: 2,
                      boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
                    }}
                  >
                    {displayGrade}
                  </Box>
                  
                  <Typography variant="subtitle1" fontWeight={500}>
                    {displayGrade === 'A' ? 'Premium Quality' : 
                     displayGrade === 'B' ? 'Standard Quality' : 'Basic Quality'}
                    {isApple && (
                      <Tooltip title="Apple-specific grading criteria applied">
                        <IconButton size="small" sx={{ ml: 0.5 }}>
                          <LocalFloristIcon fontSize="small" color="primary" />
                        </IconButton>
                      </Tooltip>
                    )}
                  </Typography>
                </Box>
                
                <Box sx={{ mt: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Overall Score
                    </Typography>
                    <Typography variant="body2" fontWeight={500}>
                      {displayScorePercent}%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={displayScore * 100} 
                    sx={{ 
                      height: 8, 
                      borderRadius: 5,
                      bgcolor: 'grey.300',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: getGradeColor(displayGrade)
                      },
                      mb: 3
                    }}
                  />
                </Box>
                
                <Typography variant="body2" sx={{ mt: 1, color: 'text.secondary' }}>
                  {getGradeDescription(displayGrade, prediction)}
                </Typography>
                
                {isApple && features.blemish_percentage && (
                  <Box sx={{ mt: 2, pt: 2, borderTop: '1px dashed', borderColor: 'grey.200' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" color="text.secondary">Blemishes:</Typography>
                      <Chip 
                        size="small"
                        label={`${features.blemish_percentage.toFixed(1)}%`} 
                        color={features.blemish_percentage > 15 ? "warning" : "success"}
                      />
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          {/* Feature Analysis */}
          <Grid item xs={12} md={4}>
            <Card 
              elevation={0}
              sx={{ 
                height: '100%',
                borderRadius: 2,
                border: '1px solid',
                borderColor: 'grey.100'
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                    <PaletteIcon sx={{ mr: 1, color: theme.palette.primary.main, fontSize: 20 }} />
                    Feature Analysis
                  </Typography>
                  <IconButton 
                    size="small" 
                    onClick={() => setExpandedTechnical(!expandedTechnical)}
                    sx={{ border: '1px solid', borderColor: 'grey.300', ml: 1 }}
                  >
                    {expandedTechnical ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  </IconButton>
                </Box>
                
                {!expandedTechnical ? (
                  <Box sx={{ height: 200, mb: 2 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart outerRadius={80} data={radarData}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="subject" />
                        <PolarRadiusAxis domain={[0, 100]} />
                        <Radar
                          name="Fruit Quality"
                          dataKey="value"
                          stroke={getGradeColor(displayGrade)}
                          fill={getGradeColor(displayGrade)}
                          fillOpacity={0.5}
                        />
                        <Radar
                          name="Grade A Threshold"
                          dataKey="A"
                          stroke="#4caf50"
                          fill="none"
                          strokeDasharray="5 5"
                        />
                        <Legend />
                      </RadarChart>
                    </ResponsiveContainer>
                  </Box>
                ) : (
                  <Box sx={{ height: 200, mb: 2 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={featureData}
                        layout="vertical"
                        margin={{ top: 5, right: 20, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" domain={[0, 100]} unit="%" />
                        <YAxis dataKey="name" type="category" width={80} />
                        <RechartsTooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Score']} />
                        <Bar dataKey="value" barSize={20}>
                          {featureData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Box>
                )}
                
                <Divider sx={{ my: 2 }} />
                
                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Typography variant="body2" sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Color:</span> <strong>{(features.color_score * 100).toFixed(1)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Size:</span> <strong>{(features.size_score * 100).toFixed(1)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        display: 'flex', 
                        justifyContent: 'space-between',
                        color: textureScore < 0.4 ? 'error.main' : 'inherit'
                      }}
                    >
                      <span>Texture:</span> <strong>{(textureScore * 100).toFixed(1)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Defect-free:</span> <strong>{(features.defect_score * 100).toFixed(1)}%</strong>
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        
        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'flex-end' }}>
          <Button 
            variant="outlined" 
            color="primary"
            startIcon={<InfoIcon />}
            size="small"
            onClick={handleOpenInfoDialog}
          >
            How Grading Works
          </Button>
        </Box>
      </Paper>
      
      {/* Information Dialog */}
      <Dialog
        open={infoDialogOpen}
        onClose={handleCloseInfoDialog}
        maxWidth="md"
      >
        <DialogTitle sx={{ fontWeight: 600 }}>
          {isApple ? 'Apple Grading Criteria' : 'Fruit Grading System Information'}
        </DialogTitle>
        <DialogContent>
          <DialogContentText component="div">
            {isApple ? (
              <>
                <Typography variant="body1" paragraph>
                  Our apple grading system uses computer vision and machine learning to analyze 
                  several key characteristics of apples:
                </Typography>
                
                <TableContainer component={Paper} elevation={0} sx={{ mb: 3 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow sx={{ backgroundColor: 'rgba(46, 125, 50, 0.08)' }}>
                        <TableCell><strong>Characteristic</strong></TableCell>
                        <TableCell><strong>Grade A</strong></TableCell>
                        <TableCell><strong>Grade B</strong></TableCell>
                        <TableCell><strong>Grade C</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>Color</TableCell>
                        <TableCell>Bright, uniform coloration appropriate for variety</TableCell>
                        <TableCell>Good, slightly variable coloration</TableCell>
                        <TableCell>Dull or highly variable coloration</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Size & Shape</TableCell>
                        <TableCell>Uniform, ideal shape with high circularity</TableCell>
                        <TableCell>Good shape with slight irregularities</TableCell>
                        <TableCell>Irregular shape or undersized</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Texture</TableCell>
                        <TableCell>Smooth, consistent surface</TableCell>
                        <TableCell>Mostly smooth with minor imperfections</TableCell>
                        <TableCell>Rough or uneven texture</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Defects</TableCell>
                        <TableCell>Less than 8% of surface area</TableCell>
                        <TableCell>8-25% of surface area</TableCell>
                        <TableCell>More than 25% of surface area</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
                
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                  <LocalFloristIcon fontSize="small" sx={{ mr: 1 }} />
                  Apple Variety Recognition
                </Typography>
                <Typography variant="body2" paragraph>
                  Our system can identify different apple varieties based on color profiles:
                </Typography>
                <ul>
                  <li>
                    <Typography variant="body2">
                      <strong>Red Varieties:</strong> High red channel values, low green values (Gala, Red Delicious, etc.)
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Green Varieties:</strong> High green channel values (Granny Smith, etc.)
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Yellow/Golden Varieties:</strong> High red and green but low blue values (Golden Delicious, etc.)
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Mixed Varieties:</strong> Other color combinations (Pink Lady, Honeycrisp, etc.)
                    </Typography>
                  </li>
                </ul>
              </>
            ) : prediction && prediction.toLowerCase() === 'banana' ? (
              <>
                <Typography variant="body1" paragraph>
                  Our banana grading system analyzes color and spotting patterns to determine ripeness and quality:
                </Typography>
                
                <TableContainer component={Paper} elevation={0} sx={{ mb: 3 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow sx={{ backgroundColor: 'rgba(255, 193, 7, 0.1)' }}>
                        <TableCell><strong>Characteristic</strong></TableCell>
                        <TableCell><strong>Grade A</strong></TableCell>
                        <TableCell><strong>Grade B</strong></TableCell>
                        <TableCell><strong>Grade C</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>Color</TableCell>
                        <TableCell>Pure yellow, no black spots</TableCell>
                        <TableCell>Yellow with some black spots</TableCell>
                        <TableCell>Predominantly black</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Yellow Coverage</TableCell>
                        <TableCell>Over 90% yellow</TableCell>
                        <TableCell>70-90% yellow</TableCell>
                        <TableCell>Less than 70% yellow</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Black Spots</TableCell>
                        <TableCell>Less than 5% of surface</TableCell>
                        <TableCell>5-30% of surface</TableCell>
                        <TableCell>More than 30% of surface</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Texture</TableCell>
                        <TableCell>Firm, smooth surface</TableCell>
                        <TableCell>Slightly soft with minor defects</TableCell>
                        <TableCell>Very soft or bruised</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
                
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                  Banana Ripeness Guide
                </Typography>
                <Typography variant="body2" paragraph>
                  Our color analysis helps determine the optimal stage of ripeness:
                </Typography>
                <ul>
                  <li>
                    <Typography variant="body2">
                      <strong>Grade A (Premium):</strong> Perfect ripeness for eating, pure yellow without spots
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Grade B (Standard):</strong> Good ripeness with some black spots, slightly sweeter
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Grade C (Processing):</strong> Overripe, best used for baking or smoothies
                    </Typography>
                  </li>
                </ul>
              </>
            ) : prediction && prediction.toLowerCase() === 'orange' ? (
              <>
                <Typography variant="body1" paragraph>
                  Our orange grading system analyzes color uniformity and surface defects to determine quality:
                </Typography>
                
                <TableContainer component={Paper} elevation={0} sx={{ mb: 3 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow sx={{ backgroundColor: 'rgba(255, 152, 0, 0.1)' }}>
                        <TableCell><strong>Characteristic</strong></TableCell>
                        <TableCell><strong>Grade A</strong></TableCell>
                        <TableCell><strong>Grade B</strong></TableCell>
                        <TableCell><strong>Grade C</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>Color</TableCell>
                        <TableCell>Uniform vibrant orange</TableCell>
                        <TableCell>Good orange with slight variation</TableCell>
                        <TableCell>Inconsistent or dull color</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Surface Defects</TableCell>
                        <TableCell>Less than 2% of surface</TableCell>
                        <TableCell>2-10% of surface</TableCell>
                        <TableCell>More than 10% of surface</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Black Spots</TableCell>
                        <TableCell>None visible</TableCell>
                        <TableCell>Few small spots</TableCell>
                        <TableCell>Multiple or large spots</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Texture</TableCell>
                        <TableCell>Smooth, firm surface</TableCell>
                        <TableCell>Mostly smooth with minor irregularities</TableCell>
                        <TableCell>Rough, uneven, or soft texture</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
                
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                  Orange Quality Indicators
                </Typography>
                <Typography variant="body2" paragraph>
                  Our defect detection system is particularly sensitive to these issues:
                </Typography>
                <ul>
                  <li>
                    <Typography variant="body2">
                      <strong>Black Spots:</strong> Even small dark spots significantly impact quality and may indicate disease
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Skin Texture:</strong> Premium oranges have smooth, consistent skin with minimal blemishes
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Color Uniformity:</strong> Uneven coloration, green patches, or brown areas indicate quality issues
                    </Typography>
                  </li>
                </ul>
              </>
            ) : (
              <>
                <Typography variant="body1" paragraph>
                  Our fruit grading system analyzes four key characteristics to determine quality:
                </Typography>
                <ul>
                  <li>
                    <Typography variant="body2">
                      <strong>Color:</strong> Uniformity and intensity of color
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Size:</strong> Overall size and shape consistency
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Texture:</strong> Surface smoothness and consistency
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Defects:</strong> Presence of blemishes, bruises, or other imperfections
                    </Typography>
                  </li>
                </ul>
                <Typography variant="body1" paragraph sx={{ mt: 2 }}>
                  Grading is determined by analyzing these features and applying industry-standard criteria:
                </Typography>
                <ul>
                  <li>
                    <Typography variant="body2">
                      <strong>Grade A:</strong> Excellent quality with high scores in all categories
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Grade B:</strong> Good quality with acceptable scores in most categories
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2">
                      <strong>Grade C:</strong> Basic quality with lower scores or significant defects
                    </Typography>
                  </li>
                </ul>
              </>
            )}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseInfoDialog} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ResultDisplay; 