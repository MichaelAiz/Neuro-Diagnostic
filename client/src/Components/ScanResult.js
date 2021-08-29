import React from 'react'

const ScanResult = ({scanClass}) => {
    let diagnosis = " "
    if(scanClass === 0){
        diagnosis = "Cognitavely Normal"
    } else if(scanClass === 1){
        diagnosis = "Mildly Cognitavely Impaired"
    } else {
        diagnosis = "Alzheimer's Disease"
    }
    return (
        <div className = "diagnosis">
            <h1 className = "diagnosisHeader">Diagnosis: {diagnosis} </h1>
        </div>
    )
}

export default ScanResult
