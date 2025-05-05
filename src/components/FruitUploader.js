import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Box, 
  Button, 
  Typography, 
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  Zoom,
  Divider,
  Card,
  CardMedia,
  CardContent,
  useMediaQuery,
  useTheme,
  Badge,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tab,
  Tabs
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import InfoIcon from '@mui/icons-material/Info';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import AppleIcon from '@mui/icons-material/Apple';
import LocalFloristIcon from '@mui/icons-material/LocalFlorist';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ImageIcon from '@mui/icons-material/Image';
import WbSunnyIcon from '@mui/icons-material/WbSunny';
import FilterCenterFocusIcon from '@mui/icons-material/FilterCenterFocus';
import CropFreeIcon from '@mui/icons-material/CropFree';
import CollectionsIcon from '@mui/icons-material/Collections';
import axios from 'axios';

const FruitUploader = ({ onImageSelect, onSubmit, supportedFruits, loading }) => {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [modelType, setModelType] = useState('auto');
  const [modelTypes, setModelTypes] = useState(['auto', 'traditional']);
  const [dragActive, setDragActive] = useState(false);
  const [selectedPhotoTab, setSelectedPhotoTab] = useState(0);
  
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Check if CNN model is available
  useEffect(() => {
    const checkModels = async () => {
      try {
        const response = await axios.get('/api/health');
        if (response.data.cnn_available) {
          setModelTypes(['auto', 'traditional', 'cnn']);
        }
      } catch (error) {
        console.error('Error checking model availability:', error);
      }
    };

    checkModels();
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    
    // Check if file is an image
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }
    
    // Create a preview URL
    const imageUrl = URL.createObjectURL(file);
    setPreviewUrl(imageUrl);
    
    // Pass the file to parent component
    onImageSelect(file);
  }, [onImageSelect]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    maxFiles: 1,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
    onDropAccepted: () => setDragActive(false)
  });

  const handleModelChange = (event) => {
    setModelType(event.target.value);
  };

  const handleSubmit = () => {
    onSubmit(modelType);
  };
  
  const handleClearImage = () => {
    setPreviewUrl(null);
    onImageSelect(null);
  };
  
  const handleTabChange = (event, newValue) => {
    setSelectedPhotoTab(newValue);
  };

  const getModelDescription = (type) => {
    switch(type) {
      case 'cnn':
        return 'Uses deep learning neural network for higher accuracy classification';
      case 'traditional':
        return 'Uses classic machine learning techniques for feature extraction';
      case 'auto':
        return 'Automatically selects the best model based on the input image';
      default:
        return '';
    }
  };

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        p: { xs: 2, md: 4 }, 
        mb: 4,
        borderRadius: 2,
        transition: 'all 0.3s ease',
        position: 'relative',
        overflow: 'hidden',
        background: 'rgba(255, 255, 255, 0.95)',
      }}
    >
      <Grid container spacing={3}>
        <Grid item xs={12} md={previewUrl ? 6 : 12}>
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
              <PhotoCameraIcon sx={{ mr: 1, color: 'primary.main' }} />
              Upload Fruit Image
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              For the most accurate results, upload a clear, well-lit image of a single fruit
            </Typography>
          </Box>
          
          <Box 
            {...getRootProps({ className: 'dropzone' })}
            sx={{ 
              p: 3, 
              border: '2px dashed', 
              borderColor: dragActive ? 'primary.main' : 'grey.300',
              borderRadius: 2,
              textAlign: 'center',
              mb: 3,
              cursor: 'pointer',
              backgroundColor: dragActive ? 'rgba(46, 125, 50, 0.04)' : 'transparent',
              transition: 'all 0.2s ease-in-out',
              '&:hover': {
                borderColor: 'primary.main',
                backgroundColor: 'rgba(46, 125, 50, 0.04)',
              },
              height: '180px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center'
            }}
          >
            <input {...getInputProps()} />
            <CloudUploadIcon sx={{ fontSize: 48, color: dragActive ? 'primary.main' : 'primary.light', mb: 1 }} />
            <Typography variant="h6" gutterBottom color={dragActive ? 'primary.main' : 'text.primary'}>
              {isDragActive ? 'Drop the image here' : 'Drag & drop a fruit image here'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              or click to select a file from your device
            </Typography>
          </Box>
          
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
              <Box component="span" sx={{ mr: 1, color: 'secondary.main' }}>🍊</Box>
              Supported Fruits
            </Typography>
            
            {supportedFruits.length > 0 ? (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                {supportedFruits.map((fruit, index) => (
                  <Chip 
                    key={index} 
                    label={fruit.charAt(0).toUpperCase() + fruit.slice(1)} 
                    color="primary" 
                    variant="outlined"
                    size="small"
                    sx={{ 
                      fontWeight: 500,
                      borderRadius: '16px'
                    }}
                    icon={fruit.toLowerCase() === 'apple' ? <AppleIcon /> : null}
                  />
                ))}
              </Box>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Loading supported fruits...
              </Typography>
            )}
          </Box>
          
          <Accordion 
            elevation={0}
            sx={{ 
              mt: 3, 
              border: '1px solid', 
              borderColor: 'primary.light', 
              borderRadius: '8px !important',
              '&:before': {
                display: 'none',
              },
              backgroundColor: 'rgba(46, 125, 50, 0.04)'
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls="model-improvement-content"
              id="model-improvement-header"
            >
              <Typography sx={{ display: 'flex', alignItems: 'center' }}>
                <CollectionsIcon sx={{ mr: 1 }} color="primary" />
                Help Improve Our Models
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" paragraph>
                To help improve our classification models, consider submitting different types of photos:
              </Typography>
              <Box sx={{ borderLeft: '3px solid', borderColor: 'primary.main', pl: 2, my: 1 }}>
                <Typography variant="body2" paragraph>
                  <strong>Different angles:</strong> Top view, side view, and multiple sides
                </Typography>
                <Typography variant="body2" paragraph>
                  <strong>Various lighting conditions:</strong> Natural daylight, indoor light, shadows
                </Typography>
                <Typography variant="body2" paragraph>
                  <strong>Different ripeness stages:</strong> Unripe, ripe, and overripe fruits
                </Typography>
                <Typography variant="body2" paragraph>
                  <strong>Multiple varieties:</strong> Different colors and shapes of the same fruit type
                </Typography>
                <Typography variant="body2" paragraph>
                  <strong>Visible defects:</strong> Fruits with blemishes, bruises, or spots help our system learn to grade accurately
                </Typography>
                <Typography variant="body2">
                  <strong>Color variations:</strong> Fruits with both uniform and non-uniform coloration to help assess quality grades
                </Typography>
              </Box>
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  These diverse images help our AI learn to:
                </Typography>
                <ul style={{ paddingLeft: '20px', margin: '8px 0' }}>
                  <li>Recognize fruits in various environments</li>
                  <li>Adapt to different photography conditions</li>
                  <li>Identify subtle features across varieties</li>
                  <li>Provide more accurate grading and classification</li>
                  <li>Detect and evaluate defects that affect quality grades</li>
                </ul>
              </Box>
            </AccordionDetails>
          </Accordion>
          
          <Accordion 
            elevation={0}
            sx={{ 
              mt: 2, 
              border: '1px solid', 
              borderColor: 'warning.light', 
              borderRadius: '8px !important',
              '&:before': {
                display: 'none',
              },
              backgroundColor: 'rgba(255, 193, 7, 0.05)'
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls="banana-grading-content"
              id="banana-grading-header"
            >
              <Typography sx={{ display: 'flex', alignItems: 'center' }}>
                <Box component="span" sx={{ mr: 1, fontSize: '1.2rem' }}>🍌</Box>
                Banana Grading Guide
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" paragraph>
                We're training our model to better grade bananas based on ripeness and quality. Banana grades are determined primarily by color:
              </Typography>
                
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={4}>
                  <Box sx={{ 
                    p: 2, 
                    bgcolor: '#fcf4a3', 
                    borderRadius: 2, 
                    height: '100%',
                    border: '1px solid #e6dc48',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center'
                  }}>
                    <Typography variant="subtitle2" fontWeight={600} align="center" gutterBottom sx={{ color: '#5f5b17' }}>
                      Grade A
                    </Typography>
                    <Box component="span" sx={{ fontSize: '2rem', mb: 1 }}>🍌</Box>
                    <Typography variant="body2" align="center" sx={{ color: '#5f5b17' }}>
                      Pure yellow, no black spots
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={4}>
                  <Box sx={{ 
                    p: 2, 
                    bgcolor: '#fce28c', 
                    borderRadius: 2, 
                    height: '100%',
                    border: '1px solid #e0c156',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center'
                  }}>
                    <Typography variant="subtitle2" fontWeight={600} align="center" gutterBottom sx={{ color: '#5f4e17' }}>
                      Grade B
                    </Typography>
                    <Box component="span" sx={{ fontSize: '2rem', mb: 1 }}>🍌</Box>
                    <Typography variant="body2" align="center" sx={{ color: '#5f4e17' }}>
                      Yellow with some black spots
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={4}>
                  <Box sx={{ 
                    p: 2, 
                    bgcolor: '#d4cfcf', 
                    borderRadius: 2, 
                    height: '100%',
                    border: '1px solid #aba6a6',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center'
                  }}>
                    <Typography variant="subtitle2" fontWeight={600} align="center" gutterBottom sx={{ color: '#333' }}>
                      Grade C
                    </Typography>
                    <Box component="span" sx={{ fontSize: '2rem', mb: 1 }}>🍌</Box>
                    <Typography variant="body2" align="center" sx={{ color: '#333' }}>
                      Predominantly black
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
                
              <Typography variant="body2" paragraph>
                <strong>Help us improve:</strong> Please upload banana images at different ripeness stages to train our model. We especially need:
              </Typography>
              <ul style={{ paddingLeft: '20px', margin: '8px 0' }}>
                <li>Perfectly yellow bananas with no spots</li>
                <li>Yellow bananas with varying degrees of black spots</li>
                <li>Overripe bananas with predominantly black coloration</li>
                <li>Green (unripe) bananas</li>
                <li>Bananas with unusual discoloration or defects</li>
              </ul>
              <Typography variant="body2" sx={{ mt: 1 }}>
                Each image helps our AI become more accurate at banana grading. Your uploads contribute directly to improving the system!
              </Typography>
            </AccordionDetails>
          </Accordion>
          
          <Accordion 
            elevation={0}
            sx={{ 
              mt: 2, 
              border: '1px solid', 
              borderColor: 'primary.light', 
              borderRadius: '8px !important',
              '&:before': {
                display: 'none',
              },
              backgroundColor: 'rgba(255, 103, 0, 0.05)'
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls="orange-grading-content"
              id="orange-grading-header"
            >
              <Typography sx={{ display: 'flex', alignItems: 'center' }}>
                <Box component="span" sx={{ mr: 1, fontSize: '1.2rem' }}>🍊</Box>
                Orange Grading Guide
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" paragraph>
                Our system grades oranges based on color uniformity and surface quality. Black spots or defects significantly impact the grade:
              </Typography>
                
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={4}>
                  <Box sx={{ 
                    p: 2, 
                    bgcolor: '#ffba5c', 
                    borderRadius: 2, 
                    height: '100%',
                    border: '1px solid #ff9c24',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center'
                  }}>
                    <Typography variant="subtitle2" fontWeight={600} align="center" gutterBottom sx={{ color: '#7d4200' }}>
                      Grade A
                    </Typography>
                    <Box component="span" sx={{ fontSize: '2rem', mb: 1 }}>🍊</Box>
                    <Typography variant="body2" align="center" sx={{ color: '#7d4200' }}>
                      Uniform orange color, no spots or defects
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={4}>
                  <Box sx={{ 
                    p: 2, 
                    bgcolor: '#ffa93d', 
                    borderRadius: 2, 
                    height: '100%',
                    border: '1px solid #ff8c00',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center'
                  }}>
                    <Typography variant="subtitle2" fontWeight={600} align="center" gutterBottom sx={{ color: '#7d4200' }}>
                      Grade B
                    </Typography>
                    <Box component="span" sx={{ fontSize: '2rem', mb: 1 }}>🍊</Box>
                    <Typography variant="body2" align="center" sx={{ color: '#7d4200' }}>
                      Good color with minor defects
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={4}>
                  <Box sx={{ 
                    p: 2, 
                    bgcolor: '#ff7d00', 
                    borderRadius: 2, 
                    height: '100%',
                    border: '1px solid #cf6700',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center'
                  }}>
                    <Typography variant="subtitle2" fontWeight={600} align="center" gutterBottom sx={{ color: '#fff' }}>
                      Grade C
                    </Typography>
                    <Box component="span" sx={{ fontSize: '2rem', mb: 1 }}>🍊</Box>
                    <Typography variant="body2" align="center" sx={{ color: '#fff' }}>
                      Black spots or significant defects
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
                
              <Typography variant="body2" paragraph>
                <strong>Important quality factors:</strong> Our model is particularly sensitive to these defects in oranges:
              </Typography>
              <ul style={{ paddingLeft: '20px', margin: '8px 0' }}>
                <li><strong>Black spots:</strong> Even small dark spots significantly reduce grade</li>
                <li><strong>Uneven coloration:</strong> Patches of green or brown lower the quality score</li>
                <li><strong>Surface texture:</strong> Rough or bumpy skin reduces the grade</li>
                <li><strong>Bruising:</strong> Soft spots or discoloration indicate poor quality</li>
                <li><strong>Mold:</strong> Any sign of mold results in immediate C grade</li>
              </ul>
              <Typography variant="body2" sx={{ mt: 1 }}>
                Help us improve by submitting orange images with a range of qualities - from perfect specimens to those with various defects.
              </Typography>
            </AccordionDetails>
          </Accordion>
        </Grid>
        
        {previewUrl && (
          <Grid item xs={12} md={6}>
            <Card 
              elevation={0} 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                border: '1px solid',
                borderColor: 'grey.100',
                borderRadius: 2,
                overflow: 'hidden'
              }}
            >
              <Box sx={{ position: 'relative' }}>
                <CardMedia
                  component="img"
                  image={previewUrl}
                  alt="Fruit preview"
                  sx={{ 
                    maxHeight: 300,
                    objectFit: 'contain',
                    backgroundColor: '#f5f5f5',
                    p: 2
                  }}
                />
                <IconButton 
                  size="small" 
                  onClick={handleClearImage}
                  sx={{ 
                    position: 'absolute', 
                    top: 8, 
                    right: 8,
                    backgroundColor: 'rgba(255,255,255,0.8)',
                    '&:hover': {
                      backgroundColor: 'rgba(255,255,255,0.9)',
                    }
                  }}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Box>
              
              <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <CheckCircleIcon color="success" sx={{ mr: 1, fontSize: 20 }} />
                  <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                    Image ready for analysis
                  </Typography>
                </Box>
                
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
                  Select Model Type:
                </Typography>
                
                <FormControl fullWidth variant="outlined" size="small" sx={{ mb: 3 }}>
                  <InputLabel id="model-type-select-label">Model Type</InputLabel>
                  <Select
                    labelId="model-type-select-label"
                    id="model-type-select"
                    value={modelType}
                    label="Model Type"
                    onChange={handleModelChange}
                  >
                    {modelTypes.map((type) => (
                      <MenuItem key={type} value={type} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Box>
                          {type.charAt(0).toUpperCase() + type.slice(1)}
                          {type === 'auto' && ' (Recommended)'}
                        </Box>
                        <Tooltip 
                          title={getModelDescription(type)} 
                          placement="right"
                          TransitionComponent={Zoom}
                          arrow
                        >
                          <InfoIcon fontSize="small" sx={{ ml: 1, color: 'text.secondary', fontSize: 16 }} />
                        </Tooltip>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={handleSubmit}
                  disabled={loading}
                  size="large"
                  fullWidth
                  sx={{ 
                    py: 1.5,
                    fontWeight: 600,
                    borderRadius: '12px',
                    mt: 'auto'
                  }}
                >
                  {loading ? 'Processing...' : 'Analyze Fruit'}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
      
      {!previewUrl && (
        <Box sx={{ mt: 4 }}>
          <Divider sx={{ mb: 3 }}>
            <Chip label="Tips for Best Results" />
          </Divider>
          
          <Box sx={{ mb: 3 }}>
            <Tabs
              value={selectedPhotoTab}
              onChange={handleTabChange}
              variant="fullWidth"
              indicatorColor="primary"
              textColor="primary"
              aria-label="photo tips tabs"
              sx={{ mb: 2 }}
            >
              <Tab icon={<FilterCenterFocusIcon />} label="Basic Tips" />
              <Tab icon={<CropFreeIcon />} label="Advanced Techniques" />
            </Tabs>
            
            {selectedPhotoTab === 0 ? (
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'grey.200', borderRadius: 2, height: '100%' }}>
                    <Typography variant="subtitle1" gutterBottom color="primary" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                      <WbSunnyIcon sx={{ mr: 1 }} fontSize="small" />
                      Good Lighting
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Ensure your fruit is well-lit with natural light for accurate color analysis. Avoid harsh shadows or overexposure.
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'grey.200', borderRadius: 2, height: '100%' }}>
                    <Typography variant="subtitle1" gutterBottom color="primary" sx={{ fontWeight: 600 }}>
                      Clear Background
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Use a solid, contrasting background to help the system identify the fruit edges. White, black, or light blue backgrounds work well.
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'grey.200', borderRadius: 2, height: '100%' }}>
                    <Typography variant="subtitle1" gutterBottom color="primary" sx={{ fontWeight: 600 }}>
                      Single Fruit
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Include only one fruit in the frame for the most accurate classification results. Center the fruit in the image.
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'primary.main', borderRadius: 2, bgcolor: 'primary.light', mt: 2, color: 'white' }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600 }}>
                      Color Uniformity Matters
                    </Typography>
                    <Typography variant="body2">
                      Fruits with clear, uniform, and vibrant colors receive higher grades. Dull, blotchy, or inconsistent coloration will decrease the grade. Our system evaluates color consistency across the entire surface as a key quality indicator.
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            ) : (
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'primary.light', borderRadius: 2, height: '100%' }}>
                    <Typography variant="subtitle1" gutterBottom color="primary" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                      <CollectionsIcon sx={{ mr: 1 }} fontSize="small" />
                      Multiple Angles
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      For most comprehensive analysis, consider uploading multiple images of the same fruit from different angles (top, side, multiple sides). This helps our system assess all aspects.
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'primary.light', borderRadius: 2, height: '100%' }}>
                    <Typography variant="subtitle1" gutterBottom color="primary" sx={{ fontWeight: 600 }}>
                      Texture Visibility
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Ensure the image clearly shows the fruit's skin texture. The system analyzes texture patterns to assess quality, so make sure surface details are visible.
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'primary.light', borderRadius: 2 }}>
                    <Typography variant="subtitle1" gutterBottom color="primary" sx={{ fontWeight: 600 }}>
                      Size Reference
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      For most accurate size grading, place the fruit beside a common object for scale reference (coin, credit card, ruler). This helps our system calibrate size measurements.
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'primary.light', borderRadius: 2, bgcolor: 'rgba(255, 152, 0, 0.04)' }}>
                    <Typography variant="subtitle1" gutterBottom color="warning.dark" sx={{ fontWeight: 600 }}>
                      Defect Visibility
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      If the fruit has black spots, bruises, cuts, or other damage, make sure these are clearly visible in the photo. These defects significantly lower the grade, and our system needs to detect them accurately.
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'primary.light', borderRadius: 2, mt: 2 }}>
                    <Typography variant="subtitle1" gutterBottom color="primary" sx={{ fontWeight: 600 }}>
                      Color Quality Guide
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Our system analyzes fruit color to determine grade:
                    </Typography>
                    <Grid container spacing={1}>
                      <Grid item xs={4}>
                        <Box sx={{ p: 1, bgcolor: 'success.light', color: 'white', borderRadius: 1, textAlign: 'center' }}>
                          <Typography variant="body2" fontWeight={500}>Grade A</Typography>
                          <Typography variant="caption">Vibrant, uniform color</Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={4}>
                        <Box sx={{ p: 1, bgcolor: 'warning.light', color: 'white', borderRadius: 1, textAlign: 'center' }}>
                          <Typography variant="body2" fontWeight={500}>Grade B</Typography>
                          <Typography variant="caption">Good, slightly varied color</Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={4}>
                        <Box sx={{ p: 1, bgcolor: 'error.light', color: 'white', borderRadius: 1, textAlign: 'center' }}>
                          <Typography variant="body2" fontWeight={500}>Grade C</Typography>
                          <Typography variant="caption">Dull or inconsistent color</Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </Box>
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'warning.light', borderRadius: 2, mt: 2 }}>
                    <Typography variant="subtitle1" gutterBottom color="warning.dark" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                      <Box component="span" sx={{ mr: 1, fontSize: '1.2rem' }}>🍌</Box>
                      Banana-Specific Color Guide
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      For bananas, our system applies special color grading criteria:
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <Box sx={{ p: 1, bgcolor: '#fcf4a3', color: '#5f5b17', borderRadius: 1, textAlign: 'center', border: '1px solid #e6dc48' }}>
                          <Typography variant="body2" fontWeight={500}>Grade A - Premium</Typography>
                          <Typography variant="caption">Pure yellow, no black spots</Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box sx={{ p: 1, bgcolor: '#fce28c', color: '#5f4e17', borderRadius: 1, textAlign: 'center', border: '1px solid #e0c156' }}>
                          <Typography variant="body2" fontWeight={500}>Grade B - Standard</Typography>
                          <Typography variant="caption">Yellow with some black spots</Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box sx={{ p: 1, bgcolor: '#d4cfcf', color: '#333', borderRadius: 1, textAlign: 'center', border: '1px solid #aba6a6' }}>
                          <Typography variant="body2" fontWeight={500}>Grade C - Processing</Typography>
                          <Typography variant="caption">Predominantly black</Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </Box>
                </Grid>
                
                <Grid item xs={12}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'primary.light', borderRadius: 2, mt: 2 }}>
                    <Typography variant="subtitle1" gutterBottom color="primary" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                      <Box component="span" sx={{ mr: 1, fontSize: '1.2rem' }}>🍊</Box>
                      Orange Quality Grading
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      For oranges, defects significantly impact grading:
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <Box sx={{ p: 1, bgcolor: '#ffba5c', color: '#7d4200', borderRadius: 1, textAlign: 'center', border: '1px solid #ff9c24' }}>
                          <Typography variant="body2" fontWeight={500}>Grade A - Premium</Typography>
                          <Typography variant="caption">No black spots, uniform color</Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box sx={{ p: 1, bgcolor: '#ffa93d', color: '#7d4200', borderRadius: 1, textAlign: 'center', border: '1px solid #ff8c00' }}>
                          <Typography variant="body2" fontWeight={500}>Grade B - Standard</Typography>
                          <Typography variant="caption">Few minor defects</Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box sx={{ p: 1, bgcolor: '#ff7d00', color: '#fff', borderRadius: 1, textAlign: 'center', border: '1px solid #cf6700' }}>
                          <Typography variant="body2" fontWeight={500}>Grade C - Processing</Typography>
                          <Typography variant="caption">Black spots or significant defects</Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </Box>
                </Grid>
              </Grid>
            )}
          </Box>
        </Box>
      )}
    </Paper>
  );
};

export default FruitUploader; 