import React, { useState } from 'react';
import './App.css';

function App() {
  const [num, setNum] = useState('1');
  const [initMethod, setInitMethod] = useState('Random');
  const [imageUrl, setImageUrl] = useState(null);
  const [isConverged, setIsConverged] = useState(false); 

  const sendRequest = (endpoint, body) => {
    return fetch(`http://localhost:5000/${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })
      .then((response) => {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
          return response.json();  // 如果是JSON响应，可能包含converged状态
        } else {
          return response.blob();  // 如果是图像响应，返回Blob
        }
      });
  };

  // Generate new dataset
  const handleGenerateDataset = () => {
    // No need to pass k anymore since dataset generation is independent of k
    fetch('http://localhost:5000/generate-dataset', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    })
      .then((response) => response.blob())
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        setImageUrl(url);  // Set the new image
        setIsConverged(false);  // Reset converged state
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  // Step through KMeans algorithm
  const handleStep = () => {
    if (isConverged) {
      alert("KMeans has already converged!");  // 如果已收敛，点击时弹出alert
      return;
    }

    sendRequest('step-kmeans', { k: num, initMethod })
      .then((result) => {
        if (result.converged) {
          // 设置已收敛状态为true
          setIsConverged(true);
          alert("KMeans has converged!");
        } else {
          const url = URL.createObjectURL(result);
          setImageUrl(url);  // 设置图像
        }
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  // Run to convergence
  const handleConverge = () => {
    sendRequest('run-kmeans', { k: num, initMethod })
      .then((result) => {
        const url = URL.createObjectURL(result);  // 处理图像
        setImageUrl(url);
        setIsConverged(true);  // 标记为已收敛
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  // Reset KMeans
  const handleReset = () => {
    sendRequest('reset-kmeans', { k: num, initMethod })
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        setImageUrl(url);
        setIsConverged(false);  // 重置状态
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>KMeans Clustering Algorithm</h1>
      </header>

      <div className='body'>
        <div className='input'>
          <label htmlFor='nums'>Number of Clusters (k):</label>
          <input
            id='nums'
            type='number'
            value={num}
            onChange={(e) => setNum(e.target.value)}
            min='1'
          />

          <label htmlFor='init'>Initialization Method: </label>
          <select
            id='init'
            value={initMethod}
            onChange={(e) => setInitMethod(e.target.value)}
          >
            <option value='Random'>Random</option>
            <option value='Farthest'>Farthest First</option>
            <option value='KMeans++'>KMeans++</option>
            <option value='Manual'>Manual</option>
          </select>
        </div>

        <div className='button'>
          <button onClick={handleStep}>Step Through KMeans</button>
          <button onClick={handleConverge}>Run to Convergence</button>
          <button onClick={handleGenerateDataset}>Generate New Dataset</button>
          <button onClick={handleReset}>Reset Algorithm</button>
        </div>

        {imageUrl && <div className='result'><img src={imageUrl} alt='Kmeans Result' /></div>}
      </div>
    </div>
  );
}

export default App;