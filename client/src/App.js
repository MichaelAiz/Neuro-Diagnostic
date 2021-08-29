import React, { useState } from 'react'
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link
} from "react-router-dom";
import Homepage from './Components/Homepage'
import ScanResult from "./Components/ScanResult";
<compo></compo>

function App() {
  const[scanClass, updateScanClass] = useState("0")


  return (
    <Router>
      <Switch>
        <Route exact path="/">
          <Homepage updateScan = {updateScanClass} />
        </Route>
        <Route exact path = "/scanResult">
          <ScanResult scanClass={scanClass} />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
