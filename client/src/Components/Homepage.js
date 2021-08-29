
import React, { useState } from 'react'
import { useHistory } from "react-router-dom"
import Button from 'react-bootstrap/Button'
import Spinner from 'react-bootstrap/Spinner'
import '../index.css'
import 'axios'
import axios from 'axios'

const Homepage = ({ updateScan }) => {
    const [file, setFile] = useState("")
    const [btnClicked, clickBtn] = useState(false)
    let history = useHistory();

    const sendFile = (e) => {
        e.preventDefault()
        clickBtn(true)
        const data = new FormData()
        data.append('file', file)
        axios
            .post('http://localhost:5000/scan', data)
            .then((res) => {
                console.log(res.data);
                updateScan(res.data);
                history.push("/scanResult")
            })
            .catch((err) => alert("File Error"));
    };

    return (
        <div className = "homePage">
            <h1 className = "appHeader">
                Neuro Diagnostic
            </h1>
            <form className = "fileForm">
                <input 
                    type="file" name="file" autoComplete="off" required
                    onChange={(e) => setFile(e.target.files[0])}
                    className = "fileSelect"
                />
                    {btnClicked ? <Spinner animation = "border"></Spinner> :<Button type="submit" className="btn btn-success" onClick={sendFile}>Scan</Button>} 
            </form>

        </div>
    )
}



export default Homepage


