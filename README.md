# Tennis Analysis

This repository contains a Computer Vision project that analyses and obtain data from a tennis match.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Example](#example)

## Introduction

The project is designed to provide insights into tennis matches by analyzing video footage. The project utilizes YOLO to track both players and the ball and CNNs to find the court keypoints.

## Features

- Player detection and tracking
- Ball detection and tracking
- Calculation of movement speed for players and the ball
- Visualization of tracking data on the video

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/kaelkkd/tennis-analysis.git
   cd tennis-analysis

2. Install the requirements:
   ```sh
   gpip install requirements.txt

3. Change the input at ```ìnput_videos``` and the players heights at ```constants``` as desired

## Example
![Screenshot from 2024-06-25 23-34-48](https://github.com/kaelkkd/tennis-analysis-ml/assets/102829149/bab2a87a-cd50-49a6-af39-1d497db595b0)
