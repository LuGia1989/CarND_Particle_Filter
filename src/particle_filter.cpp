/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  
  //Create Gaussian distributions for initial x, y, theta
  // Select the Number of particles to draw
  num_particles = 100;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  std::default_random_engine generator;

 
  // Initialize particle
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(generator);
    particle.y = dist_y(generator);
    particle.theta = dist_theta(generator);
    particle.weight = 1.0;

    //Add particle and its weight to the list of particles and weights
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  
  is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  
  //Setup Gaussian noise
  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0, std_pos[2]);
  std::default_random_engine generator;

  
  for (int i = 0; i < particles.size(); ++i) {
    if (yaw_rate != 0) {
      particles[i].x += noise_x(generator) + (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += noise_y(generator) + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += noise_theta(generator) + yaw_rate * delta_t;      
    } else {
      particles[i].x += noise_x(generator) + velocity * cos(particles[i].theta) * delta_t;
      particles[i].y += noise_y(generator) + velocity * sin(particles[i].theta) * delta_t;
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  
  for (int i = 0; i < num_particles; ++i) {
    Particle &particle = particles[i];
    
    // Since the observations are given in the vehicle's coordinate and particle are in map coordinate, 
    // transform observations to map coordinate
    std::vector<LandmarkObs> xformed_observations;
    for (int j = 0; j < observations.size(); ++j) {
      LandmarkObs obs = observations[j];
      LandmarkObs xformed_obs;
      xformed_obs.x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
      xformed_obs.y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;
      xformed_observations.push_back(xformed_obs);
    }
    
    // predicted mesurements
    std::vector<LandmarkObs> predictions;
    for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
      Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range) {
        LandmarkObs pred = {landmark.id_i, landmark.x_f, landmark.y_f};
        predictions.push_back(pred);
      }
    }
    //find nearest neighbor
    for (int i = 0; i < xformed_observations.size(); ++i) {
        LandmarkObs& obs = xformed_observations[i];
        double min_dist = std::numeric_limits<double>::max();
        for (int j = 0; j < predictions.size(); ++j) {
          LandmarkObs pred = predictions[j];
          double distance = dist(obs.x, obs.y, pred.x, pred.y);
          if (distance < min_dist) {
            min_dist = distance;
            //update id
            obs.id = j;
          }
        }
      }
    
    // update particle weight
    particle.weight = 1;
    for (int l = 0; l < xformed_observations.size(); ++l) {
      LandmarkObs obs = xformed_observations[l];
      LandmarkObs pred = predictions[obs.id];
      
      double covar_xy = std_landmark[0] * std_landmark[1];
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double denom = 1 / (2 * M_PI * covar_xy);

      // Multivariate Gaussians Probability Density Function
      double dx2 = pow(obs.x - pred.x, 2);
      double dy2 = pow(obs.y - pred.y, 2);	
      double gaussian_pdf = exp(-((dx2/2*std_x*std_x) + (dy2/2*std_y*std_y))) * denom;
      
      particle.weight *= gaussian_pdf;
    }
    weights[i] = particle.weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //    // Select the Number of particles to draw
	  num_particles = 100;
  
  std::default_random_engine generator;
  std::discrete_distribution<> weights_pmf(weights.begin(), weights.end());
  std::vector<Particle> resample_particles;
  
  for (int i = 0; i < num_particles; ++i) {
    resample_particles.push_back(particles[weights_pmf(generator)]);
  }
  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
  
  particle.associations.clear();
  
  particle.sense_y.clear();
  particle.sense_x.clear();
  
  particle.associations= associations;
  particle.sense_y = sense_y;
  particle.sense_x = sense_x;
  
  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  std::vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  std::vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  std::vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
