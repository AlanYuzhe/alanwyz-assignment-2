import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [num, setNum] = useState('3');  // 默认簇数
  const [initMethod, setInitMethod] = useState('Manual');  // 默认初始化方法
  const [imageUrl, setImageUrl] = useState(null);
  const [centroids, setCentroids] = useState([]);  // 存储手动选择的质心
  const [dataPoints, setDataPoints] = useState([]);
  const [axisLimits, setAxisLimits] = useState(null);
  const canvasRef = useRef(null);
  const [isConverged, setIsConverged] = useState(false);  // 标识KMeans是否已收敛

  // 确保在数据点或质心变化时更新画布
  useEffect(() => {
    if (dataPoints.length > 0) {
      drawCanvas();
    }
  }, [dataPoints, centroids]);

  const handleCanvasClick = (event) => {
    if (initMethod !== 'Manual') return;
    if (centroids.length >= parseInt(num)) return;  // 不允许超过k个质心

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const canvasX = event.clientX - rect.left;
    const canvasY = event.clientY - rect.top;

    const { x, y } = canvasToData(canvasX, canvasY);

    const newCentroid = { x, y };
    const updatedCentroids = [...centroids, newCentroid];
    setCentroids(updatedCentroids);

    console.log(`Selected centroids: ${updatedCentroids.length} / ${num}`);

    if (updatedCentroids.length === parseInt(num)) {
      // 当选择的质心达到k个时，发送质心给后端
      sendManualCentroids(updatedCentroids);
    } else {
      // 否则只是显示已经选中的质心
      drawCanvas();
    }
  };

  const sendManualCentroids = (centroidsToSend) => {
    fetch('http://localhost:5000/manual-kmeans', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        centroids: centroidsToSend,  // 传递选择的质心点
        k: num,  // k个簇
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data.message);
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  function dataToCanvas(x, y) {
    const canvas = canvasRef.current;
    const { x_min, x_max, y_min, y_max } = axisLimits;
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    const x_ratio = (x - x_min) / (x_max - x_min);
    const y_ratio = (y - y_min) / (y_max - y_min);

    const canvasX = x_ratio * canvasWidth;
    const canvasY = (1 - y_ratio) * canvasHeight; // 反转y轴

    return { x: canvasX, y: canvasY };
  }

  function canvasToData(canvasX, canvasY) {
    const canvas = canvasRef.current;
    const { x_min, x_max, y_min, y_max } = axisLimits;
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    const x_ratio = canvasX / canvasWidth;
    const y_ratio = 1 - (canvasY / canvasHeight); // 反转y轴

    const x = x_ratio * (x_max - x_min) + x_min;
    const y = y_ratio * (y_max - y_min) + y_min;

    return { x, y };
  }

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !axisLimits) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 绘制数据点
    dataPoints.forEach(([x, y]) => {
      const { x: canvasX, y: canvasY } = dataToCanvas(x, y);
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 2, 0, 2 * Math.PI);
      ctx.fillStyle = 'blue';
      ctx.fill();
    });

    // 绘制质心
    centroids.forEach(({ x, y }) => {
      const { x: canvasX, y: canvasY } = dataToCanvas(x, y);
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
    });
  };

  // 生成新的数据集
  const handleGenerateDataset = () => {
    fetch('http://localhost:5000/generate-dataset', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    })
      .then((response) => response.json())
      .then((data) => {
        setDataPoints(data.data_points);
        setAxisLimits({
          x_min: data.x_min,
          x_max: data.x_max,
          y_min: data.y_min,
          y_max: data.y_max
        });
        setCentroids([]);  // 清除之前的质心
        setIsConverged(false);  // 重置收敛状态
        setImageUrl(null);  // 清除之前的图片
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

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

  // Step through KMeans algorithm
  const handleStep = () => {
    if (initMethod === 'Manual' && centroids.length !== parseInt(num)) {
      alert(`You must select exactly ${num} centroids before proceeding!`);
      return;
    }

    if (isConverged) {
      alert("KMeans has already converged!");
      return;
    }

    const body = { k: num, initMethod };
    if (initMethod === 'Manual') {
      body.centroids = centroids;
    }

    sendRequest('step-kmeans', body)
      .then((result) => {
        if (result instanceof Blob) {
          const url = URL.createObjectURL(result);
          setImageUrl(url);
        } else if (result.converged) {
          setIsConverged(true);
          alert("KMeans has converged!");
        } else if (result.error) {
          alert(result.error);
        }
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  // Run to convergence
  const handleConverge = () => {
    if (initMethod === 'Manual' && centroids.length !== parseInt(num)) {
      alert(`You must select exactly ${num} centroids before proceeding!`);
      return;
    }

    const body = { k: num, initMethod };
    if (initMethod === 'Manual') {
      body.centroids = centroids;
    }

    sendRequest('run-kmeans', body)
      .then((result) => {
        if (result instanceof Blob) {
          const url = URL.createObjectURL(result);
          setImageUrl(url);
          setIsConverged(true);
        } else if (result.error) {
          alert(result.error);
        }
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  // Reset KMeans
  const handleReset = () => {
    const body = {
      k: num,
      initMethod
    };

    if (initMethod === 'Manual') {
      setCentroids([]);  // 重置质心
    }

    sendRequest('reset-kmeans', body)
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
        <h2>KMeans Clustering Algorithm</h2>
      </header>

      <div className='body'>
        <div className='input'>
          <label htmlFor='nums'>Number of Clusters (k):</label>
          <input
            id='nums'
            type='number'
            value={num}
            onChange={(e) => {
              setNum(e.target.value);
              setCentroids([]);  // 当k值改变时，重置质心
            }}
            min='1'
          />

          <label htmlFor='init'>Initialization Method: </label>
          <select
            id='init'
            value={initMethod}
            onChange={(e) => {
              setInitMethod(e.target.value);
              setCentroids([]);  // 当初始化方法改变时，重置质心
            }}
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

        {imageUrl && (
          <div className='result'>
            <img src={imageUrl} alt='KMeans Result' />
          </div>
        )}
        
        <canvas
          id="canvas"
          width={600}
          height={400}
          onClick={handleCanvasClick}
          ref={canvasRef}
          style={{ border: '1px solid black' }}
        ></canvas>

      </div>
    </div>
  );
}

export default App;