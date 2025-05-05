import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  CircularProgress,
  AppBar,
  Toolbar,
  Paper,
  useMediaQuery,
  CssBaseline,
  Snackbar,
  Alert,
  Fade
} from '@mui/material';
import { createTheme, ThemeProvider, responsiveFontSizes } from '@mui/material/styles';
import FruitUploader from './components/FruitUploader';
import ResultDisplay from './components/ResultDisplay';
import axios from 'axios';

// Create a custom theme
let theme = createTheme({
  palette: {
    primary: {
      main: '#2e7d32', // Green shade for fruit theme
      light: '#60ad5e',
      dark: '#005005',
    },
    secondary: {
      main: '#ff9800', // Orange for fruit theme
      light: '#ffc947',
      dark: '#c66900',
    },
    background: {
      default: '#f8f9fa',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Poppins", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h4: {
      fontWeight: 600,
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 12px 0 rgba(0,0,0,0.05)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px 0 rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

// Make typography responsive
theme = responsiveFontSizes(theme);

function App() {
  const [loading, setLoading] = useState(false);
  const [image, setImage] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [fruitClasses, setFruitClasses] = useState([]);
  const [modelInfo, setModelInfo] = useState({
    traditional: false,
    cnn: false
  });
  const [snackbarOpen, setSnackbarOpen] = useState(false);

  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Fetch the available fruit classes and model info on component mount
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        // Get model info first
        const healthResponse = await axios.get('/api/health');
        setModelInfo({
          traditional: healthResponse.data.traditional_model_loaded,
          cnn: healthResponse.data.cnn_model_loaded,
          cnn_available: healthResponse.data.cnn_available
        });
        
        // Then get available classes
        const classResponse = await axios.get('/api/classes');
        setFruitClasses(classResponse.data.classes);
      } catch (error) {
        console.error('Error fetching fruit classes:', error);
        setError('Failed to connect to the server. Please ensure the API is running.');
        setSnackbarOpen(true);
      }
    };

    fetchInitialData();
  }, []);

  const handleImageUpload = (imageFile) => {
    setImage(imageFile);
    setResults(null);
    setError(null);
  };

  const handleSubmit = async (modelType = 'auto') => {
    if (!image) return;

    setLoading(true);
    setError(null);

    try {
      // Convert image to base64
      const reader = new FileReader();
      reader.readAsDataURL(image);
      reader.onloadend = async () => {
        const base64Image = reader.result;
        
        try {
          // Send the image to the API with model type
          const response = await axios.post('/api/predict', {
            image: base64Image,
            model_type: modelType
          });
          
          setResults(response.data);
        } catch (error) {
          console.error('Error classifying image:', error);
          setError(error.response?.data?.error || 'Failed to classify the image. Please try again.');
          setSnackbarOpen(true);
        } finally {
          setLoading(false);
        }
      };
    } catch (error) {
      console.error('Error processing image:', error);
      setError('Failed to process the image. Please try again.');
      setLoading(false);
      setSnackbarOpen(true);
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <AppBar position="sticky" color="primary" elevation={0}>
          <Toolbar>
            <Typography 
              variant="h6" 
              component="div" 
              sx={{ 
                flexGrow: 1, 
                display: 'flex', 
                alignItems: 'center', 
                fontWeight: 600,
                letterSpacing: '0.5px' 
              }}
            >
              <Box 
                component="span" 
                sx={{ 
                  display: 'inline-flex', 
                  mr: 1,
                  fontSize: '1.5rem'
                }}
              >
                🍎
              </Box>
              Fruit Classification & Grading
            </Typography>
          </Toolbar>
        </AppBar>
        
        <Box 
          sx={{ 
            flexGrow: 1, 
            py: 4, 
            px: isMobile ? 2 : 0,
            backgroundImage: 'linear-gradient(to bottom, rgba(46, 125, 50, 0.05), rgba(255, 152, 0, 0.05))',
          }}
        >
          <Container maxWidth="lg">
            <Paper 
              elevation={0} 
              sx={{ 
                p: { xs: 3, md: 4 }, 
                mb: 4, 
                borderRadius: 2,
                background: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(10px)',
              }}
            >
              <Box sx={{ textAlign: 'center', mb: 3 }}>
                <Typography 
                  variant="h4" 
                  component="h1" 
                  gutterBottom 
                  sx={{ 
                    fontWeight: 700,
                    background: 'linear-gradient(45deg, #2e7d32 30%, #ff9800 90%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    mb: 2
                  }}
                >
                  Multi-Fruit Classification & Grading
                </Typography>
                <Typography 
                  variant="body1" 
                  color="text.secondary" 
                  sx={{ 
                    maxWidth: '700px', 
                    mx: 'auto',
                    fontSize: '1.1rem'
                  }}
                >
                  Upload an image of a fruit to classify it and determine its quality grade.
                  {modelInfo.cnn && 
                    <Box component="span" sx={{ color: 'secondary.main', fontWeight: 500 }}>
                      {' '}CNN model available for enhanced accuracy!
                    </Box>
                  }
                </Typography>
              </Box>
            </Paper>
            
            <Fade in={true} timeout={800}>
              <Box>
                <FruitUploader 
                  onImageSelect={handleImageUpload} 
                  onSubmit={handleSubmit}
                  supportedFruits={fruitClasses}
                  loading={loading}
                />
              </Box>
            </Fade>
            
            {loading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
                <CircularProgress color="secondary" />
              </Box>
            )}
            
            {results && (
              <Fade in={true} timeout={1000}>
                <Box>
                  <ResultDisplay results={results} image={image} />
                </Box>
              </Fade>
            )}
          </Container>
        </Box>
        
        <Box 
          component="footer" 
          sx={{ 
            py: 3, 
            textAlign: 'center', 
            bgcolor: 'background.paper',
            borderTop: '1px solid rgba(0, 0, 0, 0.06)'
          }}
        >
          <Typography variant="body2" color="text.secondary">
            {results && results.model_type === 'cnn' 
              ? 'Powered by ResNet18 CNN Transfer Learning'
              : 'Powered by Machine Learning Feature Extraction'
            }
          </Typography>
        </Box>
      </Box>
      
      <Snackbar 
        open={snackbarOpen} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App; 