import './App.css'
import React, { useState } from 'react';
import axios from 'axios';
import { useEffect } from 'react';

function App() {
  const [inputValue, setInputValue] = useState('');
  const [imageURL, setImageURL] = useState('');
  const [image, setImage] = useState('')

  useEffect(() => {
    setImage("backend/0plot_image.png")
    console.log('Image URL has changed:', image);
  }, [imageURL]);

  useEffect(() => {
    setImageURL("backend/plot_image.png");
    console.log('Image URL has changed:', imageURL);
    
  }, [imageURL]);

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      const response = await axios.post('http://127.0.0.1:5000/generate_plot', {
        text: inputValue,
      });

      // Assuming the API returns the image URL
      setImageURL(response.data.imageUrl);
      setImage(response.data.imageUrl)
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleInputChange = (event) => {
    
    setInputValue(event.target.value);
  };


  return (
    <> 
    <h1 style={{ fontFamily: 'Garamond, serif' }}>Flood Level Predictor</h1>
    <p style={{ fontFamily: 'Helvetica, sans-serif' }}>
        Flood Level Predictor will take in input of flood level of each road. Some roads will have missing flood level data. This value is predicted using a combination of 2 methods - <br />
        1. Training GNN on flood level data of previous days to predict missing values in the future <br />
        2. Using inverse distance weighting method to predict missing values of a day by inversely weighting the flood levels of other roads by its distance from current road<br />
        We have used the beautiful city of Vasco, Goa to simulate our program.
    </p>
    <img src="src\assets\hackenzahirez.png" alt="img1" className='image1'/>
      <form onSubmit={handleSubmit}>
        <input placeholder="Water Levels" className='text-input' onChange={handleInputChange} value={inputValue} type='text'/>
        <button type="submit">Submit</button>
      </form>
    <br />
    <br />
    <img src= {image} alt='Please submit'/>
    <br />
    <img src= {imageURL} alt='Please submit'/>
    </>
  )
}

export default App
