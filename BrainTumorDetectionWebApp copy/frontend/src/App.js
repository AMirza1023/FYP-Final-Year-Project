import logo from './dataset-cover.jpg';
import './App.css';
import React, { useState, useEffect } from "react";
import axios from 'axios';


function App() {






  return (
    <div className="App">

      <header className="AppHeader">
        <img src={logo} className="AppLogo" alt="logo" />
      </header>


      <body>
<h1 className="textCenter">Brain Tumor Classification Model with Deep and GLCM Features</h1>
<h3 className="textSuccess"></h3>
<h4 className="alertInfo"></h4>
<label form="formFileMultiple" className="formLabel">Upload Brain MRI image:</label>
<form  action="/" method="post" enctype="multipart/formData">
    <input name="imgFile" className="formControl" type="file" />
    <input type="submit" className="btn btnDark" value="Submit Image for Prediction"></input>
</form>

<form action="/pred" method="post" enctype="multipart/formData">

    <input type="submit" className="btn btnDark" value="Start Prediction"></input>
</form>

</body>



    </div>
  );
}

export default App;

