<div align="center">

# Team BingPot!

**Camera-only race car localization**

<img src="assets/bingpot.webp" alt="Jake Peralta from Brooklyn Nine-Nine with a fist out, subtitled: Bing-- pot!" width="420">

<sub><i>Our filter, the moment the start position locks in: "Bing-- pot!"</i><br>
(Bingo + jackpot, from <a href="https://brooklyn99.fandom.com/wiki/File:Holt_Bingpot.gif">Brooklyn Nine-Nine</a>)</sub>

</div>

<br>

Estimate the pose of a race car (`x`, `y`, `yaw` in the track frame) from its onboard cameras and its speed and yaw-rate
signals, at camera rate (10 Hz), with no initial pose. Built for the Constructor Labs challenge.

<br>

<table>
  <tr>
    <td width="50%" valign="top" align="center">
      <h2>Vidya Sagar Vepa</h2>
      <sub>M.Sc. Automotive Engineering, RWTH Aachen</sub>
      <p>Applies deep learning and automation to physical systems.<br>
      Experience from Johnson Electric (R&amp;I) and Ola Electric (testing and validation).</p>
      <p><i>Open to master thesis roles.</i></p>
      <p><a href="https://linkedin.com/in/vidya-sagar-vepa">LinkedIn</a></p>
    </td>
    <td width="50%" valign="top" align="center">
      <h2>Shubhangee Singhal</h2>
      <sub>M.Sc. Automotive Engineering, RWTH Aachen</sub>
      <p>Physics-grounded, data-driven engineering for vehicle systems.<br>
      Experience in powertrain calibration (Maruti Suzuki) and MBSE research (RWTH IMSE).</p>
      <p><i>Open to internships and master thesis roles.</i></p>
      <p><a href="https://linkedin.com/in/shubhangee-singhal">LinkedIn</a></p>
    </td>
  </tr>
</table>

<br>

## For the jury

To localise your own recording with our trained model, follow **[JURY_README.md](JURY_README.md)**: download the model,
install the requirements, run one command with the path to your `.mcap`, and get the pose CSV.

## Reproduce our results

To train the model yourself from the organisers' training recording and re-create the validation score, follow
**[REPRODUCE.md](REPRODUCE.md)**.
