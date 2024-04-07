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
    <p className='desc'>
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
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
