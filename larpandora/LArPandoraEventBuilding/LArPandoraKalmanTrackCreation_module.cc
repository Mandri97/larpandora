/**
 *  @file   larpandora/LArPandoraEventBuilding/LArPandoraKalmanTrackCreation_module.cc
 *
 *  @brief  module for lar pandora track creation using Kalman filter
 */

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArObjects/LArPfoObjects.h"
#include "larpandoracontent/LArUtility/KalmanFilter.h"

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"

#include "fhiclcpp/ParameterSet.h"

#include "lardataobj/RecoBase/Track.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/TrackHitMeta.h"
#include "lardataobj/RecoBase/PFParticle.h"
#include "lardataobj/RecoBase/SpacePoint.h"
#include "lardataobj/RecoBase/Vertex.h"

#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"

#include "art/Persistency/Common/PtrMaker.h"

#include "canvas/Utilities/InputTag.h"

#include "larcore/Geometry/Geometry.h"

#include "lardata/Utilities/AssociationUtil.h"

#include "messagefacility/MessageLogger/MessageLogger.h"

#include "larpandora/LArPandoraInterface/Detectors/GetDetectorType.h"
#include "larpandora/LArPandoraInterface/Detectors/LArPandoraDetectorType.h"
#include "larpandora/LArPandoraInterface/LArPandoraHelper.h"

#include <iostream>
#include <memory>

namespace lar_pandora {

  class PandoraKalmanTrackCreation : public art::EDProducer {
  public:
    explicit PandoraKalmanTrackCreation(fhicl::ParameterSet const& pset);

    PandoraKalmanTrackCreation(PandoraKalmanTrackCreation const&) = delete;
    PandoraKalmanTrackCreation(PandoraKalmanTrackCreation&&) = delete;
    PandoraKalmanTrackCreation& operator=(PandoraKalmanTrackCreation const&) = delete;
    PandoraKalmanTrackCreation& operator=(PandoraKalmanTrackCreation&&) = delete;

    void produce(art::Event& evt) override;

  private:
    /**
     *  @brief Build a recob::Track object
     *
     *  @param id the id code for the track
     *  @param trackStateVector the vector of trajectory points for this track
     */
    recob::Track BuildTrack(const int id,
                            const lar_content::LArTrackStateVector& trackStateVector) const;
    
    std::string m_pfParticleLabel;                ///< The pf particle label

    // Kalman filter parameters
    double       m_processVariance;
    double       m_measurementVariance;
    double       m_initialCovariance;
    double       m_dt;
    unsigned int m_minTrajectoryPoints;           ///< The minimum number of trajectory points
    bool         m_useAllParticles;               ///< Build a recob::Track for every recob::PFParticle
  };

  DEFINE_ART_MODULE(PandoraKalmanTrackCreation)

} // namespace lar_pandora

//------------------------------------------------------------------------------------------------------------------------------------------
// implementation follows

namespace lar_pandora {

  PandoraKalmanTrackCreation::PandoraKalmanTrackCreation(fhicl::ParameterSet const& pset)
    : EDProducer{pset}
    , m_pfParticleLabel(pset.get<std::string>("PFParticleLabel"))
    , m_processVariance(pset.get<double>("processVariance", 1.0))
    , m_measurementVariance(pset.get<double>("measurementVariance", 1.0))
    , m_initialCovariance(pset.get<double>("initialCovariance", 1.0))
    , m_dt(pset.get<double>("dt", 1.0))
    , m_minTrajectoryPoints(pset.get<unsigned int>("MinTrajectoryPoints", 2))
    , m_useAllParticles(pset.get<bool>("UseAllParticles", false))
  {
    produces<std::vector<recob::Track>>();
    produces<art::Assns<recob::PFParticle, recob::Track>>();
    produces<art::Assns<recob::Track, recob::Hit>>();
    produces<art::Assns<recob::Track, recob::Hit, recob::TrackHitMeta>>();

    if (m_minTrajectoryPoints < 2)
      throw cet::exception("LArPandoraKalmanTrackCreation")
        << "MinTrajectoryPoints should not be smaller than 2!";
  }

  //------------------------------------------------------------------------------------------------------------------------------------------

  void PandoraKalmanTrackCreation::produce(art::Event& evt)
  {
    std::unique_ptr<std::vector<recob::Track>> outputTracks(new std::vector<recob::Track>);
    std::unique_ptr<art::Assns<recob::PFParticle, recob::Track>> outputParticlesToTracks(
      new art::Assns<recob::PFParticle, recob::Track>);
    std::unique_ptr<art::Assns<recob::Track, recob::Hit>> outputTracksToHits(
      new art::Assns<recob::Track, recob::Hit>);
    std::unique_ptr<art::Assns<recob::Track, recob::Hit, recob::TrackHitMeta>>
      outputTracksToHitsWithMeta(new art::Assns<recob::Track, recob::Hit, recob::TrackHitMeta>);

    int trackCounter(0);
    const art::PtrMaker<recob::Track> makeTrackPtr(evt);

    // Organise inputs
    PFParticleVector pfParticleVector, extraPfParticleVector;
    PFParticlesToSpacePoints pfParticlesToSpacePoints;
    PFParticlesToClusters pfParticlesToClusters;
    LArPandoraHelper::CollectPFParticles(
      evt, m_pfParticleLabel, pfParticleVector, pfParticlesToSpacePoints);
    LArPandoraHelper::CollectPFParticles(
      evt, m_pfParticleLabel, extraPfParticleVector, pfParticlesToClusters);

    VertexVector vertexVector;
    PFParticlesToVertices pfParticlesToVertices;
    LArPandoraHelper::CollectVertices(evt, m_pfParticleLabel, vertexVector, pfParticlesToVertices);

    // lambda to convert pandora::CartesianVector -> Eigen::Matrix<double, 3, 1>
    auto convertSpacePointToPosition = [](const pandora::CartesianVector& sp){
      return Eigen::Matrix<double, 3, 1>{sp.GetX(), sp.GetY(), sp.GetZ()};
    };

    // lambda to convert Eigen::Matrix<double, 3, 1> -> pandora::CartesianVector
    auto convertPositionToCartesian = [](const lar_content::KalmanFilter3D::PositionVector &pos){
      return pandora::CartesianVector(pos.coeff(0, 0), pos.coeff(1, 0), pos.coeff(2, 0));
    };

    for (const art::Ptr<recob::PFParticle> pPFParticle : pfParticleVector) {
      // Select track-like pfparticles
      if (!m_useAllParticles && !LArPandoraHelper::IsTrack(pPFParticle)) continue;

      // Obtain associated spacepoints
      PFParticlesToSpacePoints::const_iterator particleToSpacePointIter(
        pfParticlesToSpacePoints.find(pPFParticle));

      if (pfParticlesToSpacePoints.end() == particleToSpacePointIter) {
        mf::LogDebug("PandoraKalmanTrackCreation") << "No spacepoints associated to particle ";
        continue;
      }

      // Obtain associated clusters
      PFParticlesToClusters::const_iterator particleToClustersIter(
        pfParticlesToClusters.find(pPFParticle));

      if (pfParticlesToClusters.end() == particleToClustersIter) {
        mf::LogDebug("LArPandoraShowerCreation") << "No clusters associated to particle ";
        continue;
      }

      // Obtain associated vertex
      PFParticlesToVertices::const_iterator particleToVertexIter(
        pfParticlesToVertices.find(pPFParticle));

      if ((pfParticlesToVertices.end() == particleToVertexIter) ||
          (1 != particleToVertexIter->second.size())) {
        mf::LogDebug("PandoraKalmanTrackCreation") << "Unexpected number of vertices for particle ";
        continue;
      }

      // Copy information for sorting purposes
      pandora::CartesianPointVector cartesianPointVector;
      for (const art::Ptr<recob::SpacePoint> spacePoint : particleToSpacePointIter->second)
        cartesianPointVector.emplace_back(pandora::CartesianVector(
          spacePoint->XYZ()[0], spacePoint->XYZ()[1], spacePoint->XYZ()[2]));
      
      if (cartesianPointVector.empty()) {
        throw cet::exception("PandoraKalmanTrackCreation")
          << "cartesianPointVector is empty!";
      }

      double vertexXYZ[3] = {0., 0., 0.};
      particleToVertexIter->second.front()->XYZ(vertexXYZ);
      const pandora::CartesianVector vertexPosition(vertexXYZ[0], vertexXYZ[1], vertexXYZ[2]);

      // Sort space spoints using the vertex position as starting point then by their distance relative to the last hit
      std::sort(cartesianPointVector.begin(), cartesianPointVector.end(), 
        [vertexPosition](const pandora::CartesianVector& lhs, const pandora::CartesianVector& rhs)
        { return lhs.GetDistanceSquared(vertexPosition) < rhs.GetDistanceSquared(vertexPosition); }
      );

      const unsigned int nPointVector = cartesianPointVector.size();

      pandora::CartesianPointVector sortedCartesianPointVector;
      sortedCartesianPointVector.reserve(nPointVector);

      std::vector<bool> isSorted(nPointVector, false);

      // Starting space point
      sortedCartesianPointVector.emplace_back(cartesianPointVector.at(0));
      isSorted.at(0) = true;

      unsigned int nSortedPoints = 1;

      for( unsigned int pointIndex = 1; pointIndex < nPointVector; ++pointIndex ){
        float minDistance = std::numeric_limits<float>::max();
        unsigned int nearestNeighborIndex = std::numeric_limits<unsigned int>::max();

        for( unsigned int nextPointIndex = 0; nextPointIndex < nPointVector; ++nextPointIndex ){
          if( isSorted.at(nextPointIndex) ) continue;
                  
          if( nextPointIndex != pointIndex ){
            const float distance = cartesianPointVector.at(pointIndex).GetDistanceSquared(
                                   cartesianPointVector.at(nextPointIndex));

            if( distance < minDistance ){
              minDistance = distance;
              nearestNeighborIndex = nextPointIndex;
            } 
          } else if( (nextPointIndex == pointIndex) && (nSortedPoints + 1 == nPointVector) ){
            nearestNeighborIndex = nextPointIndex;
          }
        }
          
        if( nearestNeighborIndex != std::numeric_limits<unsigned int>::max() ){
          sortedCartesianPointVector.emplace_back(cartesianPointVector.at(nearestNeighborIndex));
          isSorted.at(nearestNeighborIndex) = true;
          ++nSortedPoints;
        }
      }

      if( sortedCartesianPointVector.size() != nPointVector ){
        throw cet::exception("PandoraKalmanTrackCreation")
          << "sortedCartesianPointVector.size() is different than cartesianPointVector.size()";
      }

      // Initial position
      std::vector<pandora::CartesianVector>::const_iterator spacePointIter = sortedCartesianPointVector.begin();
      lar_content::KalmanFilter3D::PositionVector initialState{convertSpacePointToPosition(*(spacePointIter++))};

      lar_content::KalmanFilter3D fitter(
        m_dt, m_processVariance, m_measurementVariance, initialState, m_initialCovariance);
      
      pandora::IntVector indicesWithInvalidSpacePoint; 
      int index{0};

      std::vector<std::pair<int, lar_content::KalmanFilter3D::StateVector>> statesWithIndex;
      for( ; spacePointIter <= sortedCartesianPointVector.end(); spacePointIter++ ){
        fitter.Predict();

        if( spacePointIter < sortedCartesianPointVector.end() ){
          lar_content::KalmanFilter3D::MeasurementVector measurementState{convertSpacePointToPosition(*spacePointIter)};
          fitter.Update(measurementState);
        }

        lar_content::KalmanFilter3D::StateVector state{fitter.GetState()};
        if (!state.allFinite()) {
          indicesWithInvalidSpacePoint.push_back(index++);

          continue;
        }

        statesWithIndex.emplace_back(std::make_pair(index++, state));
      }

      if (statesWithIndex.size() < m_minTrajectoryPoints) {
        mf::LogDebug("PandoraKalmanTrackCreation")
          << "Insufficient input trajectory points to build track: " << statesWithIndex.size();
        continue;
      }

      std::vector<lar_content::KalmanFilter3D::StateVector> trackStateVector;
      pandora::IntVector indexVector;

      // Store sorted states and their index for association
      for( const auto& it:statesWithIndex ){
        indexVector.push_back(it.first);
        trackStateVector.push_back(it.second);
      }

      // Store indices of spacepoints with no associated trajectory point at the end of indexVector
      for( const int indexInvalid : indicesWithInvalidSpacePoint) {
        indexVector.push_back(indexInvalid);
      }

      HitVector hitsFromSpacePoints, hitsFromClusters, hitsInParticle;
      HitSet hitsInParticleSet;

      LArPandoraHelper::GetAssociatedHits(evt,
                                          m_pfParticleLabel,
                                          particleToSpacePointIter->second,
                                          hitsFromSpacePoints,
                                          &indexVector);
      LArPandoraHelper::GetAssociatedHits(
        evt, m_pfParticleLabel, particleToClustersIter->second, hitsFromClusters);

      //ATTN: hits ordered from space points if available, rest added at the end
      for (unsigned int hitIndex = 0; hitIndex < hitsFromSpacePoints.size(); hitIndex++) {
        hitsInParticle.push_back(hitsFromSpacePoints.at(hitIndex));
        (void)hitsInParticleSet.insert(hitsFromSpacePoints.at(hitIndex));
      }

      for (unsigned int hitIndex = 0; hitIndex < hitsFromClusters.size(); hitIndex++) {
        if (hitsInParticleSet.count(hitsFromClusters.at(hitIndex)) == 0)
          hitsInParticle.push_back(hitsFromClusters.at(hitIndex));
      }
      
      if (trackStateVector.size() > hitsFromSpacePoints.size()) {
        throw cet::exception("PandoraKalmanTrackCreation")
          << "trackStateVector.size() is greater than hitsFromSpacePoints.size()";
      }

      // Add invalid points at the end of the vector, so that the number of the trajectory points is the same as the number of hits
      const unsigned int nInvalidPoints = hitsInParticle.size() - trackStateVector.size();
      for (unsigned int i = 0; i < nInvalidPoints; ++i) {
        lar_content::KalmanFilter3D::StateVector tmpState;
        tmpState.head(3) = convertSpacePointToPosition(pandora::CartesianVector(util::kBogusF, util::kBogusF, util::kBogusF));
        tmpState.tail(3) = tmpState.head(3);

        trackStateVector.emplace_back(std::move(tmpState));
      }

      
      lar_content::LArTrackStateVector CartTrackStateVector;
      for (const lar_content::KalmanFilter3D::StateVector &sv : trackStateVector){
        CartTrackStateVector.emplace_back(
          lar_content::LArTrackState(
            convertPositionToCartesian(sv.head(3)),
            convertPositionToCartesian(sv.tail(3).normalized())
          )
        );
      }

      // Output objects
      outputTracks->emplace_back(
        PandoraKalmanTrackCreation::BuildTrack(trackCounter++, CartTrackStateVector));
      art::Ptr<recob::Track> pTrack(makeTrackPtr(outputTracks->size() - 1));

      // Output associations, after output objects are in place
      util::CreateAssn(evt, pTrack, pPFParticle, *(outputParticlesToTracks.get()));
      util::CreateAssn(evt, *(outputTracks.get()), hitsInParticle, *(outputTracksToHits.get()));

      //ATTN: metadata added with index from space points if available, null for others
      for (unsigned int hitIndex = 0; hitIndex < hitsInParticle.size(); hitIndex++) {
        const art::Ptr<recob::Hit> pHit(hitsInParticle.at(hitIndex));
        const int index((hitIndex < hitsFromSpacePoints.size()) ? hitIndex :
                                                                  std::numeric_limits<int>::max());
        recob::TrackHitMeta metadata(index, -std::numeric_limits<double>::max());
        outputTracksToHitsWithMeta->addSingle(pTrack, pHit, metadata);
      }
    }

    mf::LogDebug("PandoraKalmanTrackCreation")
      << "Number of new tracks: " << outputTracks->size() << std::endl;

    evt.put(std::move(outputTracks));
    evt.put(std::move(outputTracksToHits));
    evt.put(std::move(outputTracksToHitsWithMeta));
    evt.put(std::move(outputParticlesToTracks));
  }

  //------------------------------------------------------------------------------------------------------------------------------------------

  recob::Track PandoraKalmanTrackCreation::BuildTrack(
    const int id,
    const lar_content::LArTrackStateVector& trackStateVector) const
  {
    if (trackStateVector.empty())
      throw cet::exception("PandoraKalmanTrackCreation")
        << "BuildTrack - No input trajectory points provided ";

    recob::tracking::Positions_t xyz;
    recob::tracking::Momenta_t pxpypz;
    recob::TrackTrajectory::Flags_t flags;

    for (const lar_content::LArTrackState& trackState : trackStateVector) {
      xyz.emplace_back(recob::tracking::Point_t(trackState.GetPosition().GetX(),
                                                trackState.GetPosition().GetY(),
                                                trackState.GetPosition().GetZ()));
      pxpypz.emplace_back(recob::tracking::Vector_t(trackState.GetDirection().GetX(),
                                                    trackState.GetDirection().GetY(),
                                                    trackState.GetDirection().GetZ()));
      // Set flag NoPoint if point has bogus coordinates, otherwise use clean flag set
      if (std::fabs(trackState.GetPosition().GetX() - util::kBogusF) <
            std::numeric_limits<float>::epsilon() &&
          std::fabs(trackState.GetPosition().GetY() - util::kBogusF) <
            std::numeric_limits<float>::epsilon() &&
          std::fabs(trackState.GetPosition().GetZ() - util::kBogusF) <
            std::numeric_limits<float>::epsilon()) {
        flags.emplace_back(recob::TrajectoryPointFlags(recob::TrajectoryPointFlags::InvalidHitIndex,
                                                       recob::TrajectoryPointFlagTraits::NoPoint));
      }
      else {
        flags.emplace_back(recob::TrajectoryPointFlags());
      }
    }

    // note from gc: eventually we should produce a TrackTrajectory, not a Track with empty covariance matrix and bogus chi2, etc.
    return recob::Track(
      recob::TrackTrajectory(std::move(xyz), std::move(pxpypz), std::move(flags), false),
      util::kBogusI,
      util::kBogusF,
      util::kBogusI,
      recob::tracking::SMatrixSym55(),
      recob::tracking::SMatrixSym55(),
      id);
  }

} // namespace lar_pandora
