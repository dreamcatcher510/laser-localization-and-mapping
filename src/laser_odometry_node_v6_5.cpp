/**
 * @brief This version aims to optimize programing. Clear each module.
 */

#include <ros/ros.h>
#include <tr1/array>

/** ros laser scan */
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PolygonStamped.h>   // rviz polygon

/** ros transform tf */
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

/** pcl */
// conversion
//#include <pcl/point_cloud.h>      // If include <pcl_ros/point_cloud.h>,
//#include <pcl/point_types.h>      // these two can be not included.
#include <pcl_ros/point_cloud.h>    // This header allows you to publish
                                    // and subscribe pcl::PointCloud<T> objects as ROS messages.
// hull
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>

// Random Sample Consensus model for line fitting
#include <pcl/filters/conditional_removal.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_parallel_line.h>

//clustering
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// transform point cloud
#include <pcl/common/transforms.h>

// distance
#include <pcl/common/geometry.h>

// projecting points
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

// registration
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_lm.h>

// rviz marker
#include <visualization_msgs/Marker.h>

//Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>
// Levenberg Marquardt Non Linear Optimization
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

//OpenCV
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>

// isir message
#include "isir_common_msgs/DBUG_EST_ATT.h"
#include "isir_common_msgs/EARTH.h"

/**
 * @brief Generic functor Levenberg-Marquardt minimization
 * @tparam _Scalar Type of internal computation
 * @tparam NX Number of values per sample at compile time
 * @tparam NY Number of samples at compile time
 */
template<typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  // Number of values per sample
  const int m_inputs ;

  // Number of sample
  const int m_values;

  /**
  * @brief Default constructor
  */
  Functor()
    : m_inputs( InputsAtCompileTime ),
      m_values( ValuesAtCompileTime )
  {

  }

  /**
  * @brief Constructor
  * @param inputs Number of column per sample
  * @param values Number of sample
  */
  Functor( int inputs, int values ) : m_inputs( inputs ), m_values( values ) {}

  /**
  * @brief Get number of samples
  * @return Number of samples
  */
  int inputs() const
  {
    return m_inputs;
  }


  /**
  * @brief Get number of samples
  * @return Number of samples
  */
  int values() const
  {
    return m_values;
  }

  // you should define that in the subclass :
//  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

/**
* @brief Eigen Levemberg-Marquardt functor to refine translation, Rotation and Scale parameter.
*/
/** 3D rigid transformation estimation (4 dof)
 * Compute a yaw Rotation and Translation rigid transformation.
 * using the following formulation Xb = R * Xa + t, then Xb should one the plane.
 *
 * \param[in] x1 The first 3xN matrix of euclidean points
 * \param[in] x2 The second a set of plane coefficients
 * \param[in] t The initial 3x1 translation (last time)
 * \param[in] current roll and pitch from IMU
 * \param[in] yaw from last time
 * \param[out] delta t The 3x1 translation
 * \param[out] delta yaw orientation
 *
 * \return true if the transformation estimation has succeeded
 *
 * \note Need at least ? points
 */
struct lm_MotionEstimation_functor : Functor<double>
{
  /**
  * @brief Constructor
  * @param[in] inputs - Number of unknown inputs (unknown elements to calculate)
  * @param[in] values - Number of known samples
  * @param[in] cluster_inliers_clouds and graph_indices Input samples
  *        for scan points
  * @param[in] coefficients_** - Input samples for plane models
  * @param[in] r = ( roll(k+1), pitch(k+1), yaw(k) ) Rotation vector
  * @param[in] t = (x, y, z) Translation laser with respect to the ros world frame
  */
    lm_MotionEstimation_functor( int inputs, int values,
                              const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > cluster_inliers_clouds,
                              const std::vector<int>& graph_indices,
                              const pcl::ModelCoefficients::Ptr coefficients_fp,
                              const pcl::ModelCoefficients::Ptr coefficients_lp,
                              const pcl::ModelCoefficients::Ptr coefficients_bp,
                              const pcl::ModelCoefficients::Ptr coefficients_rp,
                              const Eigen::Vector3d &t, const Eigen::Vector3d &r ): Functor<double>( inputs, values ),
    cluster_clouds_( cluster_inliers_clouds ),
    graph_indices_(graph_indices),
    coef_fp_( coefficients_fp ),
    coef_lp_( coefficients_lp ),
    coef_bp_( coefficients_bp ),
    coef_rp_( coefficients_rp ),
    t_( t ),
    r_( r )
    {
        // x = {tx, ty, tz, roll, pitch, yaw}
        Eigen::Matrix3d Rx =
        ( Eigen::AngleAxis<double>( r_[0], Eigen::Vector3d::UnitX() ) ).toRotationMatrix();
        Eigen::Matrix3d Ry =
        ( Eigen::AngleAxis<double>( r_[1], Eigen::Vector3d::UnitY() ) ).toRotationMatrix();
        Eigen::Matrix3d Rz =
        ( Eigen::AngleAxis<double>( r_[2], Eigen::Vector3d::UnitZ() ) ).toRotationMatrix();
        R_f_ = Rz*Ry*Rx;
    }

    /**
    * @brief Computes error given a sample
    * @param x a Sample x = (x, y, z, dyaw) = (0, 0, 0, 0)
    * @param[out] fvec Error for each values
    */
    int operator()( const Eigen::VectorXd &x, Eigen::VectorXd &fvec ) const
    {
        // convert x to rotation matrix and a translation vector
        // x = {tx,ty,tz,yaw}
        Eigen::Vector3d transAdd = x.block<3, 1>( 0, 0 );
        double rot_z = x(3);

        //Build the rotation matrix
        Eigen::Matrix3d Rzcor =
          ( Eigen::AngleAxis<double>( rot_z, Eigen::Vector3d::UnitZ() ) ).toRotationMatrix();
        Eigen::Matrix3d Rx_pi =
          ( Eigen::AngleAxis<double>( M_PI, Eigen::Vector3d::UnitX() ) ).toRotationMatrix();

        const Eigen::Matrix3d nR  = Rx_pi * Rzcor * R_f_ * Rx_pi;
        const Eigen::Vector3d nt = t_ + transAdd;

        // Evaluate re-projection errors (point to plane)
        double d;
        double a, b, c, h;

        assert( cluster_clouds_.size() == graph_indices_.size() );

        //    ax + by + cz + h = 0
        //    a = coefficients->values[0], b = coefficients->values[1],
        //    c = coefficients->values[2], h = coefficients->values[3]
        int counter = -1;
        for (int i = 0; i < cluster_clouds_.size(); i++ )
        {
             switch (graph_indices_[i])
             {
                // front
                case 0:
                a = coef_fp_->values[0];
                b = coef_fp_->values[1];
                c = coef_fp_->values[2];
                h = coef_fp_->values[3];
                 break;
                // left
                case 1:
                a = coef_lp_->values[0];
                b = coef_lp_->values[1];
                c = coef_lp_->values[2];
                h = coef_lp_->values[3];
                 break;
                // back
                case 2:
                a = coef_bp_->values[0];
                b = coef_bp_->values[1];
                c = coef_bp_->values[2];
                h = coef_bp_->values[3];
                 break;
                // right
                case 3:
                a = coef_rp_->values[0];
                b = coef_rp_->values[1];
                c = coef_rp_->values[2];
                h = coef_rp_->values[3];
                 break;
        //                default: cout << "qwerty";
        //                    break;
             }
            for ( int j = 0; j < cluster_clouds_[i]->points.size(); j++ )
            {
                Eigen::Vector3d x_l(0, 0, 0);
                Eigen::Vector3d x_w(0, 0, 0);
                x_l << cluster_clouds_[i]->points[j].x,
                    cluster_clouds_[i]->points[j].y,
                    cluster_clouds_[i]->points[j].z;
                x_w = nR * x_l + nt;
                double t = std::sqrt(a*a + b*b + c*c);
                d = (a * x_w[0] + b * x_w[1] + c * x_w[2] + h)/t;
                counter++;
                fvec[counter] = d;
            }
        }
        return 0;
  }

    // Store data reference for cost evaluation
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > cluster_clouds_;
    std::vector<int> graph_indices_;
    pcl::ModelCoefficients::Ptr coef_fp_;
    pcl::ModelCoefficients::Ptr coef_lp_;
    pcl::ModelCoefficients::Ptr coef_bp_;
    pcl::ModelCoefficients::Ptr coef_rp_;
    Eigen::Vector3d t_;
    Eigen::Vector3d r_;
    Eigen::Matrix3d R_f_;
};

/**
* @brief Eigen Levemberg-Marquardt functor to refine translation, Rotation and Scale parameter.
*/
/** 3D rigid transformation estimation (4 dof)
 * Compute a yaw Rotation and Translation rigid transformation.
 * using the following formulation Xb = R * Xa + t, then Xb should one the plane.
 *
 * \param[in] x1 The first 3xN matrix of euclidean points
 * \param[in] x2 The second a set of plane coefficients
 * \param[in] t The initial 3x1 translation (last time)
 * \param[in] current roll and pitch from IMU
 * \param[in] yaw from last time
 * \param[out] delta t The 3x1 translation
 * \param[out] delta yaw orientation
 *
 * \return true if the transformation estimation has succeeded
 *
 * \note Need at least ? points
 */
struct lm_MultiViewEstimation_functor : Functor<double>
{
  /**
  * @brief Constructor
  * @param inputs Number of inputs (nb elements to refine)
  * @param values Number of samples
  * @param cluster_inliers_clouds and graph_indices Input samples
  *        for scan points
  * @param coefficients_** Input samples for plane models
  * @param r = ( roll(k+1), pitch(k+1), yaw(k) ) Rotation vector
  * @param t = (x, y, z) Translation laser with respect to the ros world frame
  */
    lm_MultiViewEstimation_functor( int inputs, int values,
                              const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > cluster_inliers_clouds_multi,
                              const std::vector<double> dt_multi,
                              const int i_multi,
                              const pcl::ModelCoefficients::Ptr coefficients_fp,
                              const pcl::ModelCoefficients::Ptr coefficients_lp,
                              const pcl::ModelCoefficients::Ptr coefficients_bp,
                              const pcl::ModelCoefficients::Ptr coefficients_rp,
                              const Eigen::Vector3d &t, const Eigen::Vector3d &r ): Functor<double>( inputs, values ),
    cluster_clouds_multi_( cluster_inliers_clouds_multi ),
    dt_multi_(dt_multi),
    i_multi_ (i_multi),
    coef_fp_( coefficients_fp ),
    coef_lp_( coefficients_lp ),
    coef_bp_( coefficients_bp ),
    coef_rp_( coefficients_rp ),
    t_( t ),
    r_( r )
    {
        // x = {tx, ty, tz, roll, pitch, yaw}
        Eigen::Matrix3d Rx =
        ( Eigen::AngleAxis<double>( r_[0], Eigen::Vector3d::UnitX() ) ).toRotationMatrix();
        Eigen::Matrix3d Ry =
        ( Eigen::AngleAxis<double>( r_[1], Eigen::Vector3d::UnitY() ) ).toRotationMatrix();
        Eigen::Matrix3d Rz =
        ( Eigen::AngleAxis<double>( r_[2], Eigen::Vector3d::UnitZ() ) ).toRotationMatrix();

        R_f_ = Rz*Ry*Rx;
    }

    /**
    * @brief Computes error given a sample
    * @param x a Sample x = (x, y, z, dyaw) = (0, 0, 0, 0)
    * @param[out] fvec Error for each values
    */
    int operator()( const Eigen::VectorXd &x, Eigen::VectorXd &fvec ) const
    {
        assert( cluster_clouds_multi_.size() == dt_multi_.size() );
        assert( i_multi_ < dt_multi_.size() );

        // Evaluate re-projection errors (point to plane)
        double d;
        double a, b, c, h;

        //    ax + by + cz + h = 0
        //    a = coefficients->values[0], b = coefficients->values[1],
        //    c = coefficients->values[2], h = coefficients->values[3]
        int counter = -1;

        Eigen::Matrix3d Rx_pi =
          ( Eigen::AngleAxis<double>( M_PI, Eigen::Vector3d::UnitX() ) ).toRotationMatrix();

        // multi-view loop
        for ( int i = 0; i < cluster_clouds_multi_.size(); ++i )
        {
            // the earliest index
            int index = ( i_multi_ + 1 + i) % cluster_clouds_multi_.size();

            // convert x to rotation matrix and a translation vector
            // x = {tx,ty,tz,roll, pitch, yaw}
            Eigen::Vector3d transAdd = x.block<3, 1>( 0, 0 );
            Eigen::Vector3d rotAdd = x.block<3, 1>( 3, 0 );

            // rotation matrix
            Eigen::Matrix3d Rxcor =
            ( Eigen::AngleAxis<double>( (i+1)*rotAdd[0], Eigen::Vector3d::UnitX() ) ).toRotationMatrix();
            Eigen::Matrix3d Rycor =
            ( Eigen::AngleAxis<double>( (i+1)*rotAdd[1], Eigen::Vector3d::UnitY() ) ).toRotationMatrix();
            Eigen::Matrix3d Rzcor =
            ( Eigen::AngleAxis<double>( (i+1)*rotAdd[2], Eigen::Vector3d::UnitZ() ) ).toRotationMatrix();

            Eigen::Matrix3d Rcor = Rzcor * Rycor * Rxcor;
            const Eigen::Matrix3d nR  = Rx_pi * R_f_ * Rcor * Rx_pi;

            // translation vector
            const Eigen::Vector3d nt = t_ + (i + 1) * transAdd;

            // error
            // each line segment loop
            for ( int j = 0; j < cluster_clouds_multi_[index]->points.size(); ++j )
            {
                // decide with plane
                // front - red
                if ( std::fabs (cluster_clouds_multi_[index]->points[j].r - 255) < 0.01
                     && std::fabs (cluster_clouds_multi_[index]->points[j].g - 0) < 0.01
                     && std::fabs (cluster_clouds_multi_[index]->points[j].b - 0) < 0.01 )
                {
                    a = coef_fp_->values[0];
                    b = coef_fp_->values[1];
                    c = coef_fp_->values[2];
                    h = coef_fp_->values[3];
                }
                // left - green
                if ( std::fabs (cluster_clouds_multi_[index]->points[j].r - 0) < 0.01
                     && std::fabs (cluster_clouds_multi_[index]->points[j].g - 255) < 0.01
                     && std::fabs (cluster_clouds_multi_[index]->points[j].b - 0) < 0.01 )
                {
                    a = coef_lp_->values[0];
                    b = coef_lp_->values[1];
                    c = coef_lp_->values[2];
                    h = coef_lp_->values[3];
                }
                // back - blue
                if ( std::fabs (cluster_clouds_multi_[index]->points[j].r - 0) < 0.01
                     && std::fabs (cluster_clouds_multi_[index]->points[j].g - 0) < 0.01
                     && std::fabs (cluster_clouds_multi_[index]->points[j].b - 255) < 0.01 )
                {
                    a = coef_bp_->values[0];
                    b = coef_bp_->values[1];
                    c = coef_bp_->values[2];
                    h = coef_bp_->values[3];
                }
                // right - orange
                if ( std::fabs (cluster_clouds_multi_[index]->points[j].r - 255) < 0.01
                     && std::fabs (cluster_clouds_multi_[index]->points[j].g - 126) < 0.01
                     && std::fabs (cluster_clouds_multi_[index]->points[j].b - 0) < 0.01 )
                {
                    a = coef_rp_->values[0];
                    b = coef_rp_->values[1];
                    c = coef_rp_->values[2];
                    h = coef_rp_->values[3];
                }

                // the distance
                Eigen::Vector3d x_l(0, 0, 0);
                Eigen::Vector3d x_w(0, 0, 0);
                x_l << cluster_clouds_multi_[index]->points[j].x,
                    cluster_clouds_multi_[index]->points[j].y,
                    cluster_clouds_multi_[index]->points[j].z;
                x_w = nR * x_l + nt;
                double t = std::sqrt(a*a + b*b + c*c);
                d = (a * x_w[0] + b * x_w[1] + c * x_w[2] + h)/t;
                counter++;
                fvec[counter] = d;
            }
        }
        return 0;
  }

    // Store data reference for cost evaluation
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > cluster_clouds_multi_;
    std::vector<double> dt_multi_;
    int i_multi_;
    pcl::ModelCoefficients::Ptr coef_fp_;
    pcl::ModelCoefficients::Ptr coef_lp_;
    pcl::ModelCoefficients::Ptr coef_bp_;
    pcl::ModelCoefficients::Ptr coef_rp_;
    Eigen::Vector3d t_;
    Eigen::Vector3d r_;
    Eigen::Matrix3d R_f_;
};

/**
 * @brief Eigen Levemberg-Marquardt functor for single view pose estimation for component x, y, z, yaw.
 * Compute a yaw Rotation and Translation rigid transformation. 3D rigid transformation estimation (4 dof)
 * using the following formulation Xb = R * Xa + t, then Xb should one the plane.
 *
 * \param[in] x1 The first 3xN matrix of euclidean points
 * \param[in] x2 The second a set of plane coefficients
 * \param[in] t The initial 3x1 translation (last time)
 * \param[in] current roll and pitch from IMU
 * \param[in] yaw from last time
 * \param[out] delta t The 3x1 translation
 * \param[out] delta yaw orientation
 *
 * \return true if the transformation estimation has succeeded
 *
 * \note Need at least ? points
 */
struct lm_SingleViewEstimation_xyzY_functor : Functor<double>
{
  /**
  * @brief Constructor
  * @param inputs Number of inputs (nb elements to refine)
  * @param values Number of samples
  * @param cluster_inliers_clouds and graph_indices Input samples
  *        for scan points
  * @param coefficients_** Input samples for plane models
  * @param r = ( roll(k+1), pitch(k+1), yaw(k) ) Rotation vector
  * @param t = (x, y, z) Translation laser with respect to the ros world frame
  */
    lm_SingleViewEstimation_xyzY_functor( int inputs, int values,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_front_line,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_left_line,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_back_line,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_right_line,
                const Eigen::Vector4f& param_fp,
                const Eigen::Vector4f& param_lp,
                const Eigen::Vector4f& param_bp,
                const Eigen::Vector4f& param_rp,
                const Eigen::Vector3d& t, const Eigen::Vector3d& r ): Functor<double>( inputs, values ),
                cloud_fl_(cloud_front_line),
                cloud_ll_(cloud_left_line),
                cloud_bl_(cloud_back_line),
                cloud_rl_(cloud_right_line),
                param_fp_( param_fp ),
                param_lp_( param_lp ),
                param_bp_( param_bp ),
                param_rp_( param_rp ),
                t_( t ),
                r_( r )
    {
        // x = {tx, ty, tz, roll, pitch, yaw}
        Eigen::Matrix3d Rx =
        ( Eigen::AngleAxis<double>( r_[0], Eigen::Vector3d::UnitX() ) ).toRotationMatrix();
        Eigen::Matrix3d Ry =
        ( Eigen::AngleAxis<double>( r_[1], Eigen::Vector3d::UnitY() ) ).toRotationMatrix();
        Eigen::Matrix3d Rz =
        ( Eigen::AngleAxis<double>( r_[2], Eigen::Vector3d::UnitZ() ) ).toRotationMatrix();
        R_f_ = Rz*Ry*Rx;
    }

    /**
    * @brief Computes error given a sample
    * @param x a Sample x = (x, y, z, dyaw) = (0, 0, 0, 0)
    * @param[out] fvec Error for each values
    */
    int operator()( const Eigen::VectorXd &x, Eigen::VectorXd &fvec ) const
    {
        // convert x to rotation matrix and a translation vector
        // x = {tx,ty,tz,yaw}
        Eigen::Vector3d transAdd = x.block<3, 1>( 0, 0 );
        double rot_z = x(3);

        //Build the rotation matrix
        Eigen::Matrix3d Rzcor =
          ( Eigen::AngleAxis<double>( rot_z, Eigen::Vector3d::UnitZ() ) ).toRotationMatrix();
        Eigen::Matrix3d Rx_pi =
          ( Eigen::AngleAxis<double>( M_PI, Eigen::Vector3d::UnitX() ) ).toRotationMatrix();

        const Eigen::Matrix3d nR  = Rx_pi * Rzcor * R_f_ * Rx_pi;
        const Eigen::Vector3d nt = t_ + transAdd;

        // Evaluate re-projection errors (point to plane)
        double d;
        double a, b, c, h;
        double t;

        //    ax + by + cz + h = 0
        //    a = coefficients->values[0], b = coefficients->values[1],
        //    c = coefficients->values[2], h = coefficients->values[3]
        int counter = -1;
        // front line
        a = param_fp_[0];
        b = param_fp_[1];
        c = param_fp_[2];
        h = param_fp_[3];
        t = std::sqrt(a*a + b*b + c*c);
        for ( int i = 0; i < cloud_fl_->points.size(); ++i )
        {
            Eigen::Vector3d x_l(0, 0, 0); // point in the laser frame
            Eigen::Vector3d x_w(0, 0, 0); // point in the world frame
            x_l << cloud_fl_->points[i].x,
                   cloud_fl_->points[i].y,
                   cloud_fl_->points[i].z;
            x_w = nR * x_l + nt;
            d = (a * x_w[0] + b * x_w[1] + c * x_w[2] + h)/t;
            ++counter;
            assert( counter < values() );
            fvec[counter] = d;
        }
        // left line
        a = param_lp_[0];
        b = param_lp_[1];
        c = param_lp_[2];
        h = param_lp_[3];
        t = std::sqrt(a*a + b*b + c*c);
        for ( int i = 0; i < cloud_ll_->points.size(); ++i )
        {
            Eigen::Vector3d x_l(0, 0, 0); // point in the laser frame
            Eigen::Vector3d x_w(0, 0, 0); // point in the world frame
            x_l << cloud_ll_->points[i].x,
                   cloud_ll_->points[i].y,
                   cloud_ll_->points[i].z;
            x_w = nR * x_l + nt;
            d = (a * x_w[0] + b * x_w[1] + c * x_w[2] + h)/t;
            ++counter;
            assert( counter < values() );
            fvec[counter] = d;
        }
        // back line
        a = param_bp_[0];
        b = param_bp_[1];
        c = param_bp_[2];
        h = param_bp_[3];
        t = std::sqrt(a*a + b*b + c*c);
        for ( int i = 0; i < cloud_bl_->points.size(); ++i )
        {
            Eigen::Vector3d x_l(0, 0, 0); // point in the laser frame
            Eigen::Vector3d x_w(0, 0, 0); // point in the world frame
            x_l << cloud_bl_->points[i].x,
                   cloud_bl_->points[i].y,
                   cloud_bl_->points[i].z;
            x_w = nR * x_l + nt;
            d = (a * x_w[0] + b * x_w[1] + c * x_w[2] + h)/t;
            ++counter;
            assert( counter < values() );
            fvec[counter] = d;
        }
        // right line
        a = param_rp_[0];
        b = param_rp_[1];
        c = param_rp_[2];
        h = param_rp_[3];
        t = std::sqrt(a*a + b*b + c*c);
        for ( int i = 0; i < cloud_rl_->points.size(); ++i )
        {
            Eigen::Vector3d x_l(0, 0, 0); // point in the laser frame
            Eigen::Vector3d x_w(0, 0, 0); // point in the world frame
            x_l << cloud_rl_->points[i].x,
                   cloud_rl_->points[i].y,
                   cloud_rl_->points[i].z;
            x_w = nR * x_l + nt;
            d = (a * x_w[0] + b * x_w[1] + c * x_w[2] + h)/t;
            ++counter;
            assert( counter < values() );
            fvec[counter] = d;
        }
        return 0;
    }

    // Store data reference for cost evaluation
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_fl_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ll_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bl_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rl_;
    Eigen::Vector4f param_fp_;
    Eigen::Vector4f param_lp_;
    Eigen::Vector4f param_bp_;
    Eigen::Vector4f param_rp_;
    Eigen::Vector3d t_;
    Eigen::Vector3d r_;
    Eigen::Matrix3d R_f_;
};



class LaserOdometry
{
public:
    LaserOdometry(int que_length);
    ~LaserOdometry();
    // debaug
    void debugTest();

private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    ros::Subscriber scan_sub_;
    ros::Subscriber imu_sub_;
    ros::Subscriber att_sub_;
    ros::Subscriber alt_sub_;

    // publishers
    ros::Publisher cloud_in_pub_;
    ros::Publisher concave_polygon_pub_;
    ros::Publisher convex_polygon_pub_;
    ros::Publisher convex_vertices_pub_;
    ros::Publisher convex_vertices_pub2_;
    ros::Publisher line_list_pub_;
    ros::Publisher line_list_pub2_;
    ros::Publisher line_list_pub4_;
    ros::Publisher line_list_pub5_;
    ros::Publisher clusters_pub_;
    ros::Publisher clusters_refined_pub_;
    ros::Publisher cloud_world_all_pub_;    // all scan
    ros::Publisher cloud_world_pub_;        // registrated using geometry method
    ros::Publisher cloud_world_fail_pub_;   // geometry method fail
    ros::Publisher cloud_bar_pub_;
    ros::Publisher clusters_inliers_pub_; // inliers of the clustered cloud without identification

    ros::Publisher object_features_pub_;

    // compare geometry estimation and registration based estimation
    ros::Publisher cloud_geo_pub_;
    ros::Publisher cloud_reg_pub_; 

    ros::Publisher front_plane_pub_;
    ros::Publisher left_plane_pub_;
    ros::Publisher back_plane_pub_;
    ros::Publisher right_plane_pub_;

    tf2_ros::TransformBroadcaster tf_broadcaster_;
    std::string worldROS_frame_;
    std::string inertial_frame_;
    std::string imu_frame_;
    std::string laser_frame_;

    /** member values */
    // algorithm
    double init_time_;      // initial time when the node created
    double time_lasted_;    // lasted time from initial to current
    double time_scan_cur_;  // current scan time
    double time_scan_last_; // last scan time
    float  M_GRAVITY_;
	
    // system state (not state machine yet)
    bool b_system_inited_;              // system start
    bool b_geometry_estimation_;        // do geometry estimation ok
    bool b_geometry_estimation_ok_;     // geometry estimation ok
    bool b_plane_model_ok_;             // plane model ok
    bool b_registration_estimation_;    // do registration estimation 
    bool b_registration_estimation_ok_; // registration estimation ok

    // geometry for laser frame
    int imu_que_length_;
    int imu_counter_;   // for array needs a counter as a pointer
    std::vector<double> imu_time_;
    // imu
    std::vector<double> yaw_;   // from imu directly
    std::vector<double> pitch_;
    std::vector<double> roll_;
    std::vector<double> a_x_;   // from imu directly
    std::vector<double> a_y_;
    std::vector<double> a_z_;
    std::vector<double> v_x_;   // intergrating from imu
    std::vector<double> v_y_;
    std::vector<double> v_z_;
    std::vector<double> x_;     // twice intergration from imu
    std::vector<double> y_;
    std::vector<double> z_;
    // isir estimated attitude
    int att_counter_;
    std::vector<double> att_time_;
    std::vector<double> gamma_0_;
    std::vector<double> gamma_1_;
    std::vector<double> gamma_2_;
    std::vector<double> quat_0_;
    std::vector<double> quat_1_;
    std::vector<double> quat_2_;
    std::vector<double> quat_3_;
    // altitude
    int alt_counter_;
//    std::tr1::array<double, 100> altitude_;
    std::vector<double> altitude_;

    // laser
    int scan_counter_;
    std::vector<double> scan_time_;
    std::vector<float> delta_graph_;
    int n_assem_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_worldROS_all_;  // all cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_worldROS_;  // cloud from geometry method
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_worldROS_fail_; // cloud when geometry method fail
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_worldROS_refine_; // refined using MLA
    int scan_to_plane_counter_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_front_plane_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_left_plane_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_back_plane_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_right_plane_;
    Eigen::Vector4f param_front_plane_;
    Eigen::Vector4f param_left_plane_;
    Eigen::Vector4f param_back_plane_;
    Eigen::Vector4f param_right_plane_;

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_front_bar_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_left_bar_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_back_bar_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_right_bar_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_bar_;

    std::vector<double> inertial_roll_imu_;
    std::vector<double> inertial_pitch_imu_;
    std::vector<double> inertial_yaw_imu_;
    std::vector<double> worldROS_x_laser_;
    std::vector<double> worldROS_y_laser_;
    std::vector<double> worldROS_z_laser_;
    std::vector<double> worldROS_x_laser_f_;
    std::vector<double> worldROS_y_laser_f_;
    std::vector<double> worldROS_z_laser_f_;

    Eigen::Affine3f wROS_T_laser_last_;

    // motion estimation
    // define a counter for restored estimated states (the length of state vector)
    std::vector<double> inertial_roll_;
    std::vector<double> inertial_pitch_;
    std::vector<double> inertial_yaw_;
    std::vector<double> inertial_x_;
    std::vector<double> inertial_y_;
    std::vector<double> inertial_z_;

    // motion estimation
    // geometry method
    // z, roll, pitch, yaw fromt front line
    std::vector<double> x_geo_fl_;
    std::vector<double> y_geo_fl_;
    std::vector<double> z_geo_fl_;
    std::vector<double> roll_geo_fl_;
    std::vector<double> pitch_geo_fl_;
    std::vector<double> yaw_geo_fl_;
    // x, y, z, roll, pitch, yaw 
    std::vector<float> x_geo_;
    std::vector<float> y_geo_;
    std::vector<float> z_geo_;
    std::vector<float> roll_geo_;
    std::vector<float> pitch_geo_;
    std::vector<float> yaw_geo_;
    // single view registration method (LMA estimation)
    std::vector<double> x_singleReg_;
    std::vector<double> y_singleReg_;
    std::vector<double> z_singleReg_;
    std::vector<double> roll_singleReg_;
    std::vector<double> pitch_singleReg_;
    std::vector<double> yaw_singleReg_;
    // fused estimation
    std::vector<double> x_fusion_;
    std::vector<double> y_fusion_;
    std::vector<double> z_fusion_;
    std::vector<double> roll_fusion_;
    std::vector<double> pitch_fusion_;
    std::vector<double> yaw_fusion_;
    // good initial estimation
    std::vector<double> w_x_l_;
    std::vector<double> w_y_l_;
    std::vector<double> w_z_l_;
    std::vector<double> i_roll_imu_;
    std::vector<double> i_pitch_imu_;
    std::vector<double> i_yaw_imu_;
    // all the estimation
    std::vector<double> w_x_l_all_;
    std::vector<double> w_y_l_all_;
    std::vector<double> w_z_l_all_;
    std::vector<double> i_roll_imu_all_;
    std::vector<double> i_pitch_imu_all_;
    std::vector<double> i_yaw_imu_all_;
    // LMA estimation
    std::vector<double> w_x_l_lm_;
    std::vector<double> w_y_l_lm_;
    std::vector<double> w_z_l_lm_;
    std::vector<double> i_roll_imu_lm_;
    std::vector<double> i_pitch_imu_lm_;
    std::vector<double> i_yaw_imu_lm_;
    // model
    pcl::ModelCoefficients::Ptr coefficients_fp_;
    pcl::ModelCoefficients::Ptr coefficients_lp_;
    pcl::ModelCoefficients::Ptr coefficients_bp_;
    pcl::ModelCoefficients::Ptr coefficients_rp_;

    // velocity estimation
    int num_multiView_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_front_multi_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_left_multi_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_back_multi_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_right_multi_;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud_multi_;
    std::vector<double> delta_t_multi_;
    std::vector<std::vector<int> > identify_multi_;

    // line tracking
    struct lineSegment_ {
      pcl::PointXYZ start;
      pcl::PointXYZ end;
      float rho;
      float dis;
      bool val;
    };
    std::vector<float> angle_histograh;

    // member function
    void imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg);
    void attLLCCallback(const isir_common_msgs::DBUG_EST_ATT& att_msg);
    void altitudeCallback(const isir_common_msgs::EARTH& msg);
    void scanCallback(
            const sensor_msgs::LaserScan::ConstPtr& scan_in);

    // low pass filter
    void lowPassFilter( const double k_LPF_vl,			// the parameter of low pass filter
                        const float expectedValue,		// the expected value
                        float &filteredValue);			// the filtered value
    float pointToPointDistance(const pcl::PointXYZ &pt1, const pcl::PointXYZ &pt2);
    float pointToLineDistance(const pcl::PointXYZ &pt1, const pcl::PointXYZ &pt2, const pcl::PointXYZ &pt);

    void sensorMsgsToPCLCloud(const sensor_msgs::LaserScan::ConstPtr& scan_in,
                              const float& range_min, const float& range_max,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in);

    void convexHull(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_convex_hull,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_convex_hull2);

    void lineSegmentDetection(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_convex_hull2,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list1,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list2);

    void lineSegmentIdentification(const std::vector<float>& delta_graph_,
                                   const pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list2,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list3,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list4,
                                   std::vector<int>& graph_indices,
                                   bool& b_front_found,
                                   bool& b_left_found,
                                   bool& b_back_found,
                                   bool& b_right_found
                                   );

    void lineSegmentIdentification2(const std::vector<float>& delta_graph_,
                                   const pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list2,
                                   pcl::PointCloud<pcl::PointXYZI>::Ptr& line_list5,
                                   bool& b_front_found,
                                   bool& b_left_found,
                                   bool& b_back_found,
                                   bool& b_right_found
                                   );

    void deBugLineList(const pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list4,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr& line_list5,
                       const bool& b_front_found,
                       const bool& b_left_found,
                       const bool& b_back_found,
                       const bool& b_right_found);

    void directCluster();

    void geometryFeatureDetectionSimple(const pcl::PointCloud<pcl::PointXYZI>::Ptr& line_list5,
                                        const bool& b_front_found,
                                        const bool& b_left_found,
                                        const bool& b_back_found,
                                        const bool& b_right_found,
                                        Eigen::Vector4f& center_point);

    void geometryFeatureDetection(
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in,
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list4,
            std::vector<int>& graph_indices,
            std::vector<std::vector<int> >& cluster_indices,
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr >& cluster_inliers_clouds,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster_inliers,   // for visualize
            Eigen::Vector3f& front_left,
            Eigen::Vector3f& front_right,
            Eigen::Vector3f& left_back,
            Eigen::Vector3f& right_back,
            Eigen::Vector3f& center_point,
            bool& b_front_end,
            bool& b_left_back,
            bool& b_right_back,
            bool& b_center_point);

    void geometryFeatureDetection2(
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in,
            const pcl::PointCloud<pcl::PointXYZI>::Ptr& line_list5,
            const bool& b_front_found,
            const bool& b_left_found,
            const bool& b_back_found,
            const bool& b_right_found,
            Eigen::Vector4f& pt_front_left,
            Eigen::Vector4f& pt_front_right,
            Eigen::Vector4f& pt_front_middle,
            Eigen::Vector4f& pt_left_back,
            Eigen::Vector4f& pt_right_back,
            Eigen::Vector4f& pt_center,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_front_line,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_left_line,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_back_line,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_right_line);

   void geometryPoseEstimation_frontLine(const Eigen::Vector3d& imu_gamma,
                                           const double& laser_altimeter,
                                           const Eigen::Vector4f& pt_front_left,
                                           const Eigen::Vector4f& pt_front_right,
				           Eigen::Vector3f& i_rot_imu,
                                           Eigen::Vector3f& w_t_v,
					   Eigen::Matrix3f& laser_R_object_fl,
                                           Eigen::Vector3f& laser_t_object_fl);

    void geometryPoseEstimation(const Eigen::Vector3d& imu_gamma,
                                const double& laser_altimeter,
                                const Eigen::Vector4f& pt_front_left,
                                const Eigen::Vector4f& pt_front_right,
                                const Eigen::Vector4f& pt_center,
                                Eigen::Matrix3f& w_R_v,
				Eigen::Vector3f& i_rot_imu,
                                Eigen::Vector3f& w_t_v,
                                Eigen::Matrix3f& laser_R_object,
                                Eigen::Vector3f& laser_t_object,
				Eigen::Matrix3f& laser_R_object_fl,
				Eigen::Vector3f& laser_t_object_fl,
                                bool& b_rotation_ok,
                                bool& b_translation_ok);

    void lm_SingleViewEstimation_xyzY();

    void cloudRegistration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in,
			   const Eigen::Matrix3f& w_R_v, const Eigen::Vector3f& w_t_v,
			   const Eigen::Vector4f& pt_center, const std::string& worldROS_frame,
			   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_front_line,
			   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_left_line,
			   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_back_line,
			   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_right_line,
			   pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_worldROS_all,
			   pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_worldROS,
			   pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_worldROS_fail,
			   std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_front_plane,
			   std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_left_plane,
			   std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_back_plane,
			   std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_right_plane);

    void planeModelFitting(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_front_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_left_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_back_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_right_plane,
			   Eigen::Vector4f& param_front_plane, Eigen::Vector4f& param_left_plane,
			   Eigen::Vector4f& param_back_plane, Eigen::Vector4f& param_right_plane);

    void planeModelFittingTest(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_front_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_left_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_back_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_right_plane,
			   Eigen::Vector4f& param_front_plane, Eigen::Vector4f& param_left_plane,
			   Eigen::Vector4f& param_back_plane, Eigen::Vector4f& param_right_plane);

    void planeModelFittingPCL(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_front_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_left_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_back_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_right_plane,
			   Eigen::Vector4f& param_front_plane, Eigen::Vector4f& param_left_plane,
			   Eigen::Vector4f& param_back_plane, Eigen::Vector4f& param_right_plane);


    // debaug
    void printEulerAngles();

    // visualization
    void convexHullVisualization(const sensor_msgs::LaserScan::ConstPtr& scan_in,
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_convex_hull,
            geometry_msgs::PolygonStamped& convex_polygon_points);
    void geometryFeatureVisualization(const Eigen::Vector4f& front_left,
                                      const Eigen::Vector4f& front_right,
                                      const Eigen::Vector4f& left_back,
                                      const Eigen::Vector4f& right_back,
                                      const Eigen::Vector4f& center_point,
			        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_front_line,
				const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_left_line,
				const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_back_line,
				const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_right_line,
                                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr& object_feature_points,
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_clustered);
     void cloudPlaneVisualization(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_front_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_left_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_back_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_right_plane,
			   pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_planes);

     void cloudComparationVisualization(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in, const std::string& worldROS_frame,
					const double& x_geo, const double& y_geo, const double& z_geo,
					const double& roll_geo, const double& pitch_geo, const double& yaw_geo,
					const double& x_reg, const double& y_reg, const double& z_reg,
					const double& roll_reg, const double& pitch_reg, const double& yaw_reg,
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_geo,
					pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_reg);

     void planeVisualization(const Eigen::Vector4f& param_front_plane, const Eigen::Vector4f& param_left_plane,
                 const Eigen::Vector4f& param_back_plane, const Eigen::Vector4f& param_right_plane,
                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr& front_plane, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& left_plane,
                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr& back_plane, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& right_plane);
};

LaserOdometry::LaserOdometry(int que_length) :
    nh_private_("~"),
    b_system_inited_(false),
    b_geometry_estimation_(false),
    b_geometry_estimation_ok_(false),
    b_plane_model_ok_(false),
    b_registration_estimation_(false),
    b_registration_estimation_ok_(false),
    scan_counter_(-1),
    delta_graph_(8,0),
    scan_to_plane_counter_(-1),
    n_assem_(500),
    cloud_worldROS_all_(new pcl::PointCloud<pcl::PointXYZ>),
    cloud_worldROS_(new pcl::PointCloud<pcl::PointXYZ>),
    cloud_worldROS_fail_(new pcl::PointCloud<pcl::PointXYZ>),
    cloud_worldROS_refine_(new pcl::PointCloud<pcl::PointXYZ>),
    cloud_bar_(new pcl::PointCloud<pcl::PointXYZRGB>),
    coefficients_fp_(new pcl::ModelCoefficients),
    coefficients_lp_(new pcl::ModelCoefficients),
    coefficients_bp_(new pcl::ModelCoefficients),
    coefficients_rp_(new pcl::ModelCoefficients),
    imu_que_length_(que_length),
    imu_counter_(-1),
    imu_time_(que_length,0),
    yaw_(que_length,0),
    pitch_(que_length,0),
    roll_(que_length,0),
    a_x_(que_length,0),
    a_y_(que_length,0),
    a_z_(que_length,0),
    v_x_(que_length,0),
    v_y_(que_length,0),
    v_z_(que_length,0),
    x_(que_length,0),
    y_(que_length,0),
    z_(que_length,0),
    att_counter_(-1),
    att_time_(que_length,0),
    gamma_0_(que_length,0),
    gamma_1_(que_length,0),
    gamma_2_(que_length,0),
    quat_0_(que_length,0),
    quat_1_(que_length,0),
    quat_2_(que_length,0),
    quat_3_(que_length,0),
    alt_counter_(-1),
    altitude_(que_length,0),
    worldROS_frame_("world"),
    inertial_frame_("inertial"),
    laser_frame_("laser"),
    imu_frame_("imu"),
    M_GRAVITY_(9.81),
    wROS_T_laser_last_(Eigen::Affine3f::Identity()),
    num_multiView_(6),   // velocity esitmatin
    delta_t_multi_(num_multiView_,0)
{
    ROS_INFO_STREAM("creating laser odometry node ...");
    // sleep for 20 seconds to ease initialization of laser processing,
    // 20 secnods depends on laser data from the bag file and the delay of launching bag node
//    ros::Duration(20).sleep(); // launch file delay = 5;

    /** register subscribers */
    // note that it needs to declear the message type if the input of the callback is a pointer
    imu_sub_ = nh_.subscribe<sensor_msgs::Imu> ("/imu/data", 2,
                                                &LaserOdometry::imuCallback, this);
    att_sub_ = nh_.subscribe("/LlcControler/DBUG_EST_ATT", 2,
                             &LaserOdometry::attLLCCallback, this);
    alt_sub_ = nh_.subscribe("/LlcControler/EARTH", 2,
                             &LaserOdometry::altitudeCallback, this);
    scan_sub_ = nh_.subscribe<sensor_msgs::LaserScan>
            ("/scan", 2, &LaserOdometry::scanCallback, this);

    /** register publishers */
    cloud_in_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ> >("/cloud_in", 1);
    convex_polygon_pub_ = nh_.advertise<geometry_msgs::PolygonStamped>("/polygon_convex", 1);
    concave_polygon_pub_ = nh_.advertise<geometry_msgs::PolygonStamped>("/polygon_concave", 1);
    convex_vertices_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ> >("/vertices_convex", 1);
    convex_vertices_pub2_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ> >("/vertices_convex2", 1);
    line_list_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ> >("/line_list", 1);
    line_list_pub2_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ> >("/line_list2", 1);
    line_list_pub4_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ> >("/line_list4", 1);
    line_list_pub5_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZI> >("/line_list5", 1);

    // publish identified point cloud
    clusters_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/cloud_clusters", 1);
    clusters_refined_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/cloud_cluster_refined", 1);
    // publish inlier point cloud
    clusters_inliers_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ> > ("/cloud_clusters_inliers", 1);

    // assemblering in the ROS world frame
    cloud_world_all_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ> >("/cloud_world_all", 1);
    cloud_world_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ> >("/cloud_world_geometry", 1);
    cloud_world_fail_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ> >("/cloud_world_geometry_fail", 1);
    cloud_bar_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/cloud_bar", 1);

    // publish object feature points
    object_features_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("/object_features", 1);

    // compare geometry and registration estimation
    cloud_geo_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("/cloud_geo", 1);
    cloud_reg_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("/cloud_reg", 1);

    front_plane_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("/front_plane", 1);
    left_plane_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("/left_plane", 1);
    back_plane_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("/back_plane", 1);
    right_plane_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("/right_plane", 1);

    /**
     * \brief Build the structured graph. The graph includes two information:
     * (1) partition in polar angle space;
     * (2) the orientation of line segment in the corresponding seperated space.
     * This graph is built in a learning step. Here we simply set it manually.
     * The orientaion constraints (parallel and perpendicular) is used direclty in the following.
     * Output: delta_graph, note that orientation graph is used with priori knowlege.
     */
    float delta = 0.01;
//    std::vector<float> delta_graph_(8);
    delta_graph_[0] = delta_graph_[1] = 0;
    delta_graph_[2] = M_PI/2 - std::atan2(4.0,1.0);
    delta_graph_[3] = M_PI/2 + delta;
    delta_graph_[4] = -M_PI/3;
    delta_graph_[5] = M_PI/3;
    delta_graph_[6] = -M_PI/2 - delta;
    delta_graph_[7] = -M_PI/2 + std::atan2(4.0,1.0);

    // multi-view registration (n frames)
    cloud_front_multi_.resize(num_multiView_);
    cloud_left_multi_.resize(num_multiView_);
    cloud_back_multi_.resize(num_multiView_);
    cloud_right_multi_.resize(num_multiView_);
    cloud_multi_.resize(num_multiView_);
    identify_multi_.resize(num_multiView_);
}

LaserOdometry::~LaserOdometry()
{
    ROS_INFO_STREAM("Destroying laser odometry node ...");
}

float LaserOdometry::pointToPointDistance(const pcl::PointXYZ &pt1, const pcl::PointXYZ &pt2)
{
    return std::sqrt( (pt1.x-pt2.x) * (pt1.x-pt2.x) + (pt1.y-pt2.y) * (pt1.y-pt2.y) + (pt1.z-pt2.z) * (pt1.z-pt2.z));
}

/**
 * @brief 2 D: point to line distance.
 * Line defined by two points pt1 and pt2.
 * Refering Wikipedia: Distance from a point to a line
 */
float LaserOdometry::pointToLineDistance(const pcl::PointXYZ &pt1,
                                         const pcl::PointXYZ &pt2,
                                         const pcl::PointXYZ &pt)
{
    float denominator = std::sqrt( (pt2.y-pt1.y)*(pt2.y-pt1.y) +(pt2.x-pt1.x)*(pt2.x-pt1.x) );
    return std::abs( (pt2.y-pt1.y)*pt.x - (pt2.x-pt1.x)*pt.y + (pt2.x*pt1.y - pt2.y*pt1.x) )/denominator;
}

/**
 * @brief transform the coordinate from imu data in (imu) world frame
 * to laser coordinate in (laser) world frame.
 * The initial laser frame is the (laser) world frame. The initial imu frame is the (imu) world frame.
 * Since the fixed connection, the transform is through swapping.
 */
/**
* The conventions of the coordinate system
*
*    ^  x
*    |                        imu coordinate system (NED):
*    |                        roll along x-axis (pointing forward)
*    |                        pitch along y-axis (pointing to the left)
*    |                        yaw along z-axis (pointing upward)
*    |--------------> y       in Z(yaw)Y(pitch)X(roll) order
*     \
*      \
*       \
*        \
*         \
*          z
*
*  roll, pitch and yaw is calculate using ros coordinate system
*
*                ^ x   ^ z
*                |    /      lidar coordinate system (NWU):
*                |   /       roll along z-axis (pointing forward)
*                |  /        pitch along x-axis (pointing to the left)
*                | /         yaw along y-axis (pointing upward)
*  y <-----------|/          in Y(yaw)X(pitch)Z(roll) order
*/
void LaserOdometry::imuCallback(const sensor_msgs::Imu::ConstPtr &imu_msg)
{
//    ROS_INFO_STREAM("Receiving filtered imu data ...");
    double yaw, pitch, roll;
    tf::Quaternion orientation;
    /** convert from sensor_msgs::Imu::Quaternion to tf::Quaternion */
    tf::quaternionMsgToTF(imu_msg->orientation, orientation);
    tf::Matrix3x3 w_R_imu(orientation);
    tf::Matrix3x3 imu_R_laser;
    imu_R_laser.setEulerYPR(0, 0, M_PI);
    w_R_imu.getRPY(roll, pitch, yaw);
    tf::Quaternion imu_q_laser(1,0,0,0);
//    imu_R_laser.getRotation(w_q);

    imu_counter_ = (imu_counter_ + 1) % imu_que_length_;
    yaw_[imu_counter_] = yaw;
    pitch_[imu_counter_] = pitch;
    roll_[imu_counter_] = roll;
    imu_time_[imu_counter_] = imu_msg->header.stamp.toSec();

    /**
     * @brief World frame is selected as NEU, imu frame is selected as UWD.
     * We have: a_{m} = R^{T}(a_{world}+g*e_{3}).
     * a_{world} = R*a_{m} - g*e_{3}
     * Calculating the acceleration in the world frame
     */
    float accX = imu_msg->linear_acceleration.x;
    float accY = imu_msg->linear_acceleration.y;
    float accZ = imu_msg->linear_acceleration.z;
    tf::Vector3 acc_imu(accX, accY, accZ);
    tf::Vector3 w_acc;
    w_acc = w_R_imu*acc_imu + tf::Vector3(0, 0, M_GRAVITY_);

    a_x_[imu_counter_] = w_acc.x();
    a_y_[imu_counter_] = w_acc.y();
    a_z_[imu_counter_] = w_acc.z();

    int imu_counter_back = (imu_counter_ + imu_que_length_ - 1) % imu_que_length_;
    double time_diff = imu_time_[imu_counter_] - imu_time_[imu_counter_back];
//    ROS_INFO_STREAM("dt: " << time_diff );
    if (time_diff < 0.5) {

//      x_[imu_counter_] = x_[imu_counter_back] + v_x_[imu_counter_back] * time_diff
//                                + w_acc.x() * time_diff * time_diff / 2;
//      y_[imu_counter_] = y_[imu_counter_back] + v_y_[imu_counter_back] * time_diff
//                                + w_acc.y() * time_diff * time_diff / 2;
//      z_[imu_counter_] = z_[imu_counter_back] + v_z_[imu_counter_back] * time_diff
//                                + w_acc.z() * time_diff * time_diff / 2;

        x_[imu_counter_] = x_[imu_counter_back] + 0.1*v_x_[imu_counter_back] * time_diff;
        y_[imu_counter_] = y_[imu_counter_back] + 0.1*v_y_[imu_counter_back] * time_diff;
        z_[imu_counter_] = z_[imu_counter_back] + 0.1*v_z_[imu_counter_back] * time_diff;

        v_x_[imu_counter_] = v_x_[imu_counter_back] + 0.1*w_acc.x() * time_diff;
        v_y_[imu_counter_] = v_y_[imu_counter_back] + 0.1*w_acc.y() * time_diff;
        v_z_[imu_counter_] = v_z_[imu_counter_back] + 0.1*w_acc.z() * time_diff;
    }
}

void LaserOdometry::attLLCCallback(const isir_common_msgs::DBUG_EST_ATT& att_msg)
{
//    ROS_INFO_STREAM("Receiving and processing DBUG_EST_ATT message ...");
    att_counter_ = (att_counter_ + 1) % imu_que_length_;
    gamma_0_[att_counter_] = att_msg.EST_ATT_GAMMA0;
    gamma_1_[att_counter_] = att_msg.EST_ATT_GAMMA1;
    gamma_2_[att_counter_] = att_msg.EST_ATT_GAMMA2;
    quat_0_[att_counter_] = att_msg.EST_ATT_QUAT0;
    quat_1_[att_counter_] = att_msg.EST_ATT_QUAT1;
    quat_2_[att_counter_] = att_msg.EST_ATT_QUAT2;
    quat_3_[att_counter_] = att_msg.EST_ATT_QUAT3;
    att_time_[att_counter_] = att_msg.header.stamp.toSec();

//    double roll, pitch;
//    pitch = std::asin(gamma_0_[att_counter_]);
//    roll = std::atan2(gamma_1_[att_counter_],gamma_2_[att_counter_]);

//    Eigen::Matrix3f Rx, Ry, Rz, R, w_R_imu, wROS_R_imu, imu_R_laser;
//    Rx = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
//    Ry = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
//    Rz = Eigen::AngleAxisf(0, Eigen::Vector3f::UnitZ());
//    w_R_imu = Rz * Ry * Rx;
//    R = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX());
//    wROS_R_imu = R*w_R_imu;
//    imu_R_laser = R;
//    Eigen::Quaternionf w_quat_imu(wROS_R_imu);
//    Eigen::Quaternionf imu_quat_laser(imu_R_laser);
}

void LaserOdometry::altitudeCallback(const isir_common_msgs::EARTH& msg)
{
//    ROS_INFO_STREAM("Receiving and processing altitude message ...");
    alt_counter_ = (alt_counter_ + 1) % imu_que_length_;
    altitude_[alt_counter_] = msg.POS_Z/1000.0;
}

/**
 * @brief Convert sensor_msgs::LaserScan to pcl point cloud and publish
 * This part minics scan_tools/scan_to_cloud_converter/scan_to_cloud_converter.cpp/scanCallback
 * This method is advantage for selecting the scan range between range_min and range_max.
 * What waiting to do is to check the right way to use pcl::PointCloud<pcl::PointXYZ>::Ptr->resize().
 * Output: cloud scan_cloud_in
 */
void LaserOdometry::sensorMsgsToPCLCloud(const sensor_msgs::LaserScan::ConstPtr& scan_in,
                                         const float& range_min, const float& range_max,
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in)
{
    /**
     * @brief convention of laser coordinate system
     *
     *                ^ x
     *                |
     *                |
     *                |
     *                |
     *  y <-----------|
     */
    int cloud_size = scan_in->ranges.size();
    int beam_count = 0;
    for(unsigned int i = 0; i < cloud_size; i++)
    {
        float range = scan_in->ranges[i];
        if( range > range_min && range < range_max )
        {
            float angle = scan_in->angle_min + i*scan_in->angle_increment;
            scan_cloud_in->points[beam_count].x = range * std::cos(angle);
            scan_cloud_in->points[beam_count].y = range * std::sin(angle);
            scan_cloud_in->points[beam_count].z = 0;
            beam_count++;
        }
    }
    scan_cloud_in->width = beam_count;
    scan_cloud_in->height = 1;
    scan_cloud_in->resize(beam_count);
    pcl_conversions::toPCL(scan_in->header, scan_cloud_in->header);
}

/**
 * @brief Convex hull polygon extraction for object boundary.
 * @param[in]: scan_cloud_in, laser scan.
 * @param[out]: cloud_convex_hull, vertices points of convex hull boundary.
 * @param[out]: cloud_convex_hull2, flat vertices points of convex hull boundary.
 * Note that the the vertices of convex hull from pcl is in clockwise.
 */
void LaserOdometry::convexHull(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_convex_hull,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_convex_hull2)
{
    /**
     * \brief preliminary convex hull polygon extraction for boundary
     * Output: cloud_convex_hull (vertices points of convex hull)
     */
    std::vector< pcl::Vertices> hull_polygon;
    pcl::ConvexHull<pcl::PointXYZ> convex_hull;
    convex_hull.setInputCloud (scan_cloud_in);
//    convex_hull.reconstruct (*cloud_convex_hull);
    convex_hull.reconstruct (*cloud_convex_hull, hull_polygon);
    pcl::PolygonMesh hull_mesh;
    convex_hull.reconstruct(hull_mesh);
    cloud_convex_hull->header = scan_cloud_in->header;

    /**
     * \brief Refine convex hull polygon through merging the adjacent smooth line segments
     * which have approximate same orientation.
     * Accomplishment: discard the flat points.
     * Output: cloud_convex_hull2
     */
    pcl::PointXYZ pt_convex2;
    for (size_t i = 0; i < cloud_convex_hull->points.size(); ++i)
    {
        size_t i_p1 = (i + 1 + cloud_convex_hull->points.size()) % cloud_convex_hull->points.size();
        size_t i_m1 = (i - 1 + cloud_convex_hull->points.size()) % cloud_convex_hull->points.size();
        float dx = cloud_convex_hull->points[i_p1].x - cloud_convex_hull->points[i].x;
        float dy = cloud_convex_hull->points[i_p1].y - cloud_convex_hull->points[i].y;
        float k = std::atan2(dy, dx);
        dx = cloud_convex_hull->points[i].x - cloud_convex_hull->points[i_m1].x;
        dy = cloud_convex_hull->points[i].y - cloud_convex_hull->points[i_m1].y;
        float k_m1 = std::atan2(dy, dx);
        if ( std::abs(k-k_m1) > (M_PI/6) )
        {
            pt_convex2.x = cloud_convex_hull->points[i].x;
            pt_convex2.y = cloud_convex_hull->points[i].y;
            pt_convex2.z = cloud_convex_hull->points[i].z;
            cloud_convex_hull2->push_back(pt_convex2);
        }
    }
    // set the message header
    cloud_convex_hull2->header = scan_cloud_in->header;
}


/**
 * @brief for publish and visilization
 * (1) geometry_msgs/PolygonStamped message: for visualization in rviz using polygon;
 * (2) set the header of cloud_convex_hull to high light of the convex vertices in rviz
 */
void LaserOdometry::convexHullVisualization(
        const sensor_msgs::LaserScan::ConstPtr& scan_in,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_convex_hull,
        geometry_msgs::PolygonStamped& convex_polygon_points)
{
    convex_polygon_points.polygon.points.clear();
    geometry_msgs::Point32 pt_convex;
    for (size_t i = 0; i < cloud_convex_hull->points.size(); ++i)
    {
        pt_convex.x = cloud_convex_hull->points[i].x;
        pt_convex.y = cloud_convex_hull->points[i].y;
        pt_convex.z = cloud_convex_hull->points[i].z;
        convex_polygon_points.polygon.points.push_back(pt_convex);
    }
    // set the message header (the frame ID and timestamp).
    // See the TF tutorials for information on these.
    convex_polygon_points.header.frame_id = scan_in->header.frame_id;
    convex_polygon_points.header.stamp = scan_in->header.stamp;
}

/**
 * @brief Extracting line segments from the convex contour
 * Each line segment includes two vertices.
 * @param[out]: line_list1, discards short line segments, and stores the pair-wise two end points
 * (vertices) in clockwise.
 * @param[out]: line_list2, re-organizes the data structure starting from the nearest front line segment.
 */
void LaserOdometry::lineSegmentDetection(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_convex_hull2,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list1,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list2)
{
    // line_list1
    float threshold_short_line = 0.2;
    for (size_t i = 0; i < cloud_convex_hull2->points.size(); ++i)
    {
        size_t i_p1 = (i + 1 + cloud_convex_hull2->points.size()) % cloud_convex_hull2->points.size();
        float dx = cloud_convex_hull2->points[i_p1].x - cloud_convex_hull2->points[i].x;
        float dy = cloud_convex_hull2->points[i_p1].y - cloud_convex_hull2->points[i].y;
        float dis = std::sqrt(dx*dx + dy*dy);
        if ( dis > threshold_short_line )
        {
            line_list1->push_back(cloud_convex_hull2->points[i]);
            line_list1->push_back(cloud_convex_hull2->points[i_p1]);
        }
    }
    // set the message header
    line_list1->header = cloud_convex_hull2->header;

    /**
     * \brief re-order the line segments:
     * Selectng the front line segment as the (structured graph) base.
     *
     * The nearest front line segment is selected if the following conditions satisfied:
     * (1) The distance from origin to two end ponits include minimum distance;
     * (2) The line segment across the x-axis, which is decided using directions of the two end ponits from the origion.
     * Note that the line_list1 from the convex hull is sorted in clockwise starting from the left.
     */
    size_t i_min_dis;
    float dis_min = 100;
    float dis1, dis2, dis_temp;
    for (size_t i = 0; i < line_list1->points.size()/2; ++i)
    {
        dis1 = std::sqrt(line_list1->points[2*i].x*line_list1->points[2*i].x
                         + line_list1->points[2*i].y*line_list1->points[2*i].y);
        dis2 = std::sqrt(line_list1->points[2*i+1].x*line_list1->points[2*i+1].x
                         + line_list1->points[2*i+1].y*line_list1->points[2*i+1].y);
        if ( 0 == i )
        {
            dis_min = ( dis1 < dis2 ) ? dis1 : dis2;
            i_min_dis = 2*i;
        }
        else
        {
            dis_temp = ( dis1 < dis2 ) ? dis1 : dis2;
            // Note that this decision is not general.
            if ( dis_temp <= dis_min
                 && std::atan2(line_list1->points[2*i].y, line_list1->points[2*i].x) < 0
                 && std::atan2(line_list1->points[2*i+1].y, line_list1->points[2*i+1].x) > 0)
            {
                i_min_dis = 2*i;
                dis_min = dis_temp;
            }
        }
    }

    // use the front closest line segment as the base (put in the first)
    line_list2->resize(line_list1->points.size());
    for (size_t i = 0; i < line_list2->points.size(); ++i)
    {
        size_t j = (i + i_min_dis + line_list1->points.size()) % line_list1->points.size();
//        line_list2->push_back(line_list1->points[j]);
        pcl::PointXYZ p;
        line_list2->points[i].x = line_list1->points[j].x;
        line_list2->points[i].y = line_list1->points[j].y;
        line_list2->points[i].z = line_list1->points[j].z;

        if( 0 == i )
        {
            // this is to warning the wrong suppose of the order from convex hull extraction
            if (std::atan2(line_list2->points[2*i].y, line_list2->points[2*i].x) > 0
                    && std::atan2(line_list2->points[2*i+1].y, line_list2->points[2*i+1].x) < 0)
                ROS_WARN_STREAM("The nearest front line end points are not in right order ...");
        }
    }
    // set the message header
    line_list2->header = cloud_convex_hull2->header;
}

/**
 * @brief Identify the line segments in the line_list2 corresponding to the strucutred graph
 * First calculating the object frame and the transform
 * between the laser frame and the object frame.
 * Method one: only use the orientation constraints.
 * @param[out]: line_list3
 * Method two: use the structured garph (orientation and polar partition)
 * @param[out]: line_list4 and graph_indices, where line_list4->points.size() = 2*graph_indices.size()
 */
void LaserOdometry::lineSegmentIdentification(const std::vector<float>& delta_graph_,
                               const pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list2,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list3,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list4,
                               std::vector<int>& graph_indices,
                               bool& b_front_found,
                               bool& b_left_found,
                               bool& b_back_found,
                               bool& b_right_found
                               )
{
    b_front_found = false;
    b_left_found = false;
    b_back_found = false;
    b_right_found = false;
    /**
     * \brief calculate the frame of the line segments
     * Select the middle point of the nearest front line segment as the origin of the object frame.
     */
    Eigen::Transform<float,3,Eigen::Affine> t;  // transform from laser frame to the object frame
    // note that t here is calculated based on convex hull, it is used for identify line segments
    // with respect to structured graph model.
    // it needs to be refined by front inliers and the tower center.
    // later we define here t_front_center to different t_tower_center.
    if( line_list2->points.size() > 1 )
    {
        Eigen::Translation<float,3> trans((line_list2->points[0].x + line_list2->points[1].x)/2,
                (line_list2->points[0].y + line_list2->points[1].y)/2, 0);
        /**
         * Since the convex is in the clockwise from the top, for the nearest front line segment line_list2,
         * line_list2->points[1] is on the left, and line_list2->points[0] is on the right.
         * The vector (line_list2->points[1] - line_list2->points[0]) is selected as y-axis of the line list frame (the object frame).
         */
        float dy = line_list2->points[1].y - line_list2->points[0].y;
        float dx = line_list2->points[1].x - line_list2->points[0].x;
        float angle_in_radian = std::atan2(dy, dx) - M_PI/2;   // x-axis direction from y-axis direction
        Eigen::AngleAxis<float> aa(angle_in_radian, Eigen::Vector3f(0,0,1));
        t = trans*aa;
    }

    /**
     * \brief Selecting the line segments satisfying the orientation constraints.
     * Now we selected parallel or perpendicular to the nearest front line segments.
     * Later a structured graph is build and select the line segments using the structured
     * graph constraints.
     */
    /**
     * \brief method one: Selecting the line segments only satisfying the orientation constraints.
     * Output: line_list3
     */
//    pcl::PointCloud<pcl::PointXYZ>::Ptr line_list3
//            (new pcl::PointCloud<pcl::PointXYZ>(line_list2->points.size(), 1));
    line_list3->width = line_list2->points.size();
    line_list3->height = 1;
    line_list3->resize( line_list3->width * line_list3->height );
    size_t j_temp = 0;
    Eigen::Vector3f v_f;    // the orientation of the nearest front line
    float threshold_similar_line = 0.05;
    if ( line_list2->size() > 1 )
    {
        v_f << line_list2->points[0].x - line_list2->points[1].x,
                    line_list2->points[0].y - line_list2->points[1].y, 0;
        v_f.normalize();
        // add the front line segment
        line_list3->points[0].x = line_list2->points[0].x;
        line_list3->points[0].y = line_list2->points[0].y;
        line_list3->points[0].z = line_list2->points[0].z;
        line_list3->points[1].x = line_list2->points[1].x;
        line_list3->points[1].y = line_list2->points[1].y;
        line_list3->points[1].z = line_list2->points[1].z;
        ++j_temp;
    }
    for ( size_t i = 1; i < line_list2->points.size()/2; ++i )
    {
        Eigen::Vector3f v(line_list2->points[2*i].x - line_list2->points[2*i+1].x,
               line_list2->points[2*i].y - line_list2->points[2*i+1].y, 0);
        v.normalize();
        Eigen::Vector3f v_c = v_f.cross(v);
        if ( std::abs(v_f.dot(v)) < threshold_similar_line || std::abs(v_c.norm()) < threshold_similar_line )
        {
            line_list3->points[2*j_temp].x = line_list2->points[2*i].x;
            line_list3->points[2*j_temp].y = line_list2->points[2*i].y;
            line_list3->points[2*j_temp].z = line_list2->points[2*i].z;
            line_list3->points[2*j_temp+1].x = line_list2->points[2*i+1].x;
            line_list3->points[2*j_temp+1].y = line_list2->points[2*i+1].y;
            line_list3->points[2*j_temp+1].z = line_list2->points[2*i+1].z;
            ++j_temp;
        }
    }
    line_list3->resize(2*j_temp);

    /**
     * \brief method two: use the structured graph with orientation and polar angle space partition constraints.
     * Output: line_list4, graph_indices
     * line_list4->points.size() = 2*graph_indices.size();
     */
//    pcl::PointCloud<pcl::PointXYZ>::Ptr line_list4
//            (new pcl::PointCloud<pcl::PointXYZ>(line_list2->points.size(), 1));
    line_list4->width = line_list2->points.size();
    line_list4->height = 1;
    line_list4->resize( line_list4->width * line_list4->height );
    j_temp = 0;
//    Eigen::Vector3f v_f;    // the nearest front line
//    float threshold_similar_line = 0.05;
//    std::vector<int> graph_indices; // for line_list4, graph_indices.size() should be equal to j_temp*2.
    Eigen::Vector3f p_fm;   // front middle point
    // add the nearest front line to the list
    if ( line_list2->size() > 1 )
    {
        v_f << line_list2->points[0].x - line_list2->points[1].x,
                    line_list2->points[0].y - line_list2->points[1].y, 0;
        v_f.normalize();
        line_list4->points[0].x = line_list2->points[0].x;
        line_list4->points[0].y = line_list2->points[0].y;
        line_list4->points[0].z = line_list2->points[0].z;
        line_list4->points[1].x = line_list2->points[1].x;
        line_list4->points[1].y = line_list2->points[1].y;
        line_list4->points[1].z = line_list2->points[1].z;
        ++j_temp;
        // the front line
        graph_indices.push_back(0);
        p_fm << (line_list2->points[0].x + line_list2->points[1].x)/2,
                (line_list2->points[0].y + line_list2->points[1].y)/2, 0;

        b_front_found = true;
    }
    /**
     * \brief identify the line segments corresponding to the structured graph. Two steps:
     * (1) Transform the line segment vectors (or the scan points) to object frame.
     * This step is used in the seperated space identify step.
     * (2) Identify the ponit to the line segments using structured graph.
     */
    float alpha_1, alpha_2; // seperated space
    Eigen::Vector3f v;      // line segment orientation

    /** incert calculating geometry center point (coordinate)
     *  method one: find the left back or right back point, then calculate the geometry center point.
     *  The geometry center is use the front left and right back or the front right and the left back.
     *  If both the left back and the right back are detected, decide who is farer.
     */
    float dis_left = 0;
    float dis_right = 0;
    Eigen::Vector3f center_point(0, 0, 0);
    Eigen::Vector3f left_back(0, 0, 0);
    Eigen::Vector3f right_back(0, 0, 0);

    for ( size_t i = 1; i < line_list2->points.size()/2; ++i )
    {
        Eigen::Vector4f p1_l, p2_l, p1_o, p2_o; // two end points in laser frame and object frame
        p1_l << line_list2->points[2*i].x, line_list2->points[2*i].y, 0, 1;
        p2_l << line_list2->points[2*i+1].x, line_list2->points[2*i+1].y, 0, 1;
        p1_o = t.inverse()*p1_l;    // t.inverse() from object frame to the laser frame
        p2_o = t.inverse()*p2_l;
        alpha_1 = std::atan2(p1_o(1), p1_o(0));
        alpha_2 = std::atan2(p2_o(1), p2_o(0));

        v << line_list2->points[2*i].x - line_list2->points[2*i+1].x,
               line_list2->points[2*i].y - line_list2->points[2*i+1].y, 0;
        v.normalize();
        Eigen::Vector3f v_c = v_f.cross(v);
        // idenfity the line segment
        // left
        if ( alpha_1 >= delta_graph_[2] &&  alpha_1 <= delta_graph_[3]
             && alpha_2 >= delta_graph_[2] &&  alpha_2 <= delta_graph_[3]
             && std::abs(v_f.dot(v)) < threshold_similar_line )
        {
            line_list4->points[2*j_temp].x = line_list2->points[2*i].x;
            line_list4->points[2*j_temp].y = line_list2->points[2*i].y;
            line_list4->points[2*j_temp].z = line_list2->points[2*i].z;
            line_list4->points[2*j_temp+1].x = line_list2->points[2*i+1].x;
            line_list4->points[2*j_temp+1].y = line_list2->points[2*i+1].y;
            line_list4->points[2*j_temp+1].z = line_list2->points[2*i+1].z;
            ++j_temp;
            graph_indices.push_back(1);
//            ROS_INFO_STREAM("line segment as left ...");

            b_left_found = true;
            left_back[0] = line_list2->points[2*i+1].x;
            left_back[1] = line_list2->points[2*i+1].y;
            left_back[2] = line_list2->points[2*i+1].z;
            dis_left = pointToPointDistance(line_list2->points[2*i+1], line_list2->points[1]);
        }
        // back
        if ( alpha_1 >= delta_graph_[4] &&  alpha_1 <= delta_graph_[5]
             && alpha_2 >= delta_graph_[4] &&  alpha_2 <= delta_graph_[5]
             && std::abs(v_c.norm()) < threshold_similar_line )
        {
            line_list4->points[2*j_temp].x = line_list2->points[2*i].x;
            line_list4->points[2*j_temp].y = line_list2->points[2*i].y;
            line_list4->points[2*j_temp].z = line_list2->points[2*i].z;
            line_list4->points[2*j_temp+1].x = line_list2->points[2*i+1].x;
            line_list4->points[2*j_temp+1].y = line_list2->points[2*i+1].y;
            line_list4->points[2*j_temp+1].z = line_list2->points[2*i+1].z;
            ++j_temp;
            graph_indices.push_back(2);
            b_back_found = true;
//            ROS_INFO_STREAM("line segment as back ...");
        }
        // right
        if ( alpha_1 >= delta_graph_[6] &&  alpha_1 <= delta_graph_[7]
             && alpha_2 >= delta_graph_[6] &&  alpha_2 <= delta_graph_[7]
             && std::abs(v_f.dot(v)) < threshold_similar_line )
        {
            line_list4->points[2*j_temp].x = line_list2->points[2*i].x;
            line_list4->points[2*j_temp].y = line_list2->points[2*i].y;
            line_list4->points[2*j_temp].z = line_list2->points[2*i].z;
            line_list4->points[2*j_temp+1].x = line_list2->points[2*i+1].x;
            line_list4->points[2*j_temp+1].y = line_list2->points[2*i+1].y;
            line_list4->points[2*j_temp+1].z = line_list2->points[2*i+1].z;
            ++j_temp;
            graph_indices.push_back(3);
//            ROS_INFO_STREAM("line segment as right ...");

            b_right_found = true;
            right_back[0] = line_list2->points[2*i].x;
            right_back[1] = line_list2->points[2*i].y;
            right_back[2] = line_list2->points[2*i].z;
            dis_right = pointToPointDistance(line_list2->points[2*i], line_list2->points[0]);
        }
    }
    line_list4->resize(2*j_temp);
    // set the message header
//    pcl_conversions::toPCL(scan_in->header, line_list4->header);
    line_list4->header = line_list2->header;
    if (line_list4->points.size()/2 != graph_indices.size())
        ROS_WARN_STREAM("The size of line_list4 is not equal to graph_indices.");
    assert( line_list4->points.size()/2 == graph_indices.size() );
}

/**
 * @brief Identify the line segments in the line_list2 corresponding to the strucutred graph
 * In line_list2, a line segment may be divided into two. Here the same segments are merged.
 * Use the structured garph (orientation and polar partition)
 *
 * @param[out]: line_list5, intensity = 0, 1, 2, 3, representing front, left, back, and right respectively.
 */
void LaserOdometry::lineSegmentIdentification2(const std::vector<float>& delta_graph_,
                               const pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list2,
                               pcl::PointCloud<pcl::PointXYZI>::Ptr& line_list5,
                               bool& b_front_found,
                               bool& b_left_found,
                               bool& b_back_found,
                               bool& b_right_found
                               )
{
    b_front_found = false;
    b_left_found = false;
    b_back_found = false;
    b_right_found = false;

    line_list5->width = 8;
    line_list5->height = 1;
    line_list5->resize( line_list5->width * line_list5->height );
    size_t j_temp = -1;

    /**
     * \brief Build the object frame and compute the transformation from laser frame to the object frame.
     * Select the frame of the line segments: middle point and orientation of the nearest front line segment.
     */
    Eigen::Vector3f ori_front(0, 0, 0);    // the orientation of the nearest front line
    Eigen::Vector3f pt_fm(0, 0, 0);   // front middle point
    Eigen::Matrix3f l_R_o;
    Eigen::Vector3f l_t_o(0, 0, 0);

    if ( line_list2->points.size() > 1 )
    {
        /**
         * Since the convex is in the clockwise from the top, for the nearest front line segment line_list2,
         * line_list2->points[1] is on the left, and line_list2->points[0] is on the right.
         * The vector (line_list2->points[1] - line_list2->points[0]) is selected as y-axis of the line list frame (the object frame).
         */
        float dy = line_list2->points[1].y - line_list2->points[0].y;
        float dx = line_list2->points[1].x - line_list2->points[0].x;
        float angle_in_radian = std::atan2(dy, dx) - M_PI/2;   // x-axis direction from y-axis
        l_R_o = Eigen::AngleAxisf(angle_in_radian, Eigen::Vector3f::UnitZ());

        l_t_o << (line_list2->points[0].x + line_list2->points[1].x)/2,
                (line_list2->points[0].y + line_list2->points[1].y)/2, 0;

        // initial front line using the nearest front line
        line_list5->points[0].x = line_list2->points[0].x;
        line_list5->points[0].y = line_list2->points[0].y;
        line_list5->points[0].z = line_list2->points[0].z;
        line_list5->points[0].intensity = 0;
        line_list5->points[1].x = line_list2->points[1].x;
        line_list5->points[1].y = line_list2->points[1].y;
        line_list5->points[1].z = line_list2->points[1].z;
        line_list5->points[1].intensity = 0;

        ori_front << line_list2->points[0].x - line_list2->points[1].x,
                    line_list2->points[0].y - line_list2->points[1].y, 0;
        ori_front.normalize();
        pt_fm = l_t_o;
        b_front_found = true;
        ++j_temp;
    }

    /**
     * \brief identify the line segments corresponding to the structured graph. Two steps:
     * (1) Transform the line segment vectors (or the scan points) from laser frame to object frame.
     * This step is used in the seperated space identify step.
     * (2) Identify the ponit to the line segments using structured graph.
     */
    // for identification
    float alpha_1, alpha_2; // seperated space
    Eigen::Vector3f v;      // line segment orientation
    float threshold_similar_line = 0.05;

    /** incert calculating geometry center point (coordinate)
     *  method one: find the left back or right back point, then calculate the geometry center point.
     *  The geometry center is use the front left and right back or the front right and the left back.
     *  If both the left back and the right back are detected, decide who is farer.
     */
    for ( size_t i = 1; i < line_list2->points.size()/2; ++i )
    {
        assert( j_temp < line_list5->points.size()/2 );
        Eigen::Vector3f p1_l, p2_l, p1_o, p2_o;
        // points in the laser frame
        p1_l << line_list2->points[2*i].x, line_list2->points[2*i].y, 0;
        p2_l << line_list2->points[2*i+1].x, line_list2->points[2*i+1].y, 0;
        // points in the object frame
        p1_o = l_R_o.transpose() * p1_l - l_R_o.transpose()*l_t_o;
        p2_o = l_R_o.transpose() * p2_l - l_R_o.transpose()*l_t_o;

        alpha_1 = std::atan2(p1_o(1), p1_o(0));
        alpha_2 = std::atan2(p2_o(1), p2_o(0));

        v << line_list2->points[2*i].x - line_list2->points[2*i+1].x,
               line_list2->points[2*i].y - line_list2->points[2*i+1].y, 0;
        v.normalize();
        Eigen::Vector3f v_cross = ori_front.cross(v);
        // idenfity the line segment
//        // front
//        if ( std::abs( p1_o[0]) < threshold_similar_line
//             && p2_o[0] < threshold_similar_line
//             && std::abs(v_cross.norm()) < threshold_similar_line )
//        {
//            assert( b_front_found );
//            assert( j_temp >= 0 );
//            if ( line_list2->points[2*i+1].y > line_list5->points[1].y )
//            {
//                line_list5->points[1].x =  line_list2->points[2*i+1].x;
//                line_list5->points[1].y =  line_list2->points[2*i+1].y;
//                line_list5->points[1].z =  line_list2->points[2*i+1].z;
//                line_list5->points[1].intensity = 0;
//            }
//            else
//            {
//                line_list5->points[0].x =  line_list2->points[2*i].x;
//                line_list5->points[0].y =  line_list2->points[2*i].y;
//                line_list5->points[0].z =  line_list2->points[2*i].z;
//                line_list5->points[0].intensity = 0;
//            }
//        }
        // left
        if ( alpha_1 >= delta_graph_[2] &&  alpha_1 <= delta_graph_[3]
             && alpha_2 >= delta_graph_[2] &&  alpha_2 <= delta_graph_[3]
             && std::abs(ori_front.dot(v)) < threshold_similar_line )
        {
            if ( b_left_found )
            {
                line_list5->points[2*j_temp+1].x = line_list2->points[2*i+1].x;
                line_list5->points[2*j_temp+1].y = line_list2->points[2*i+1].y;
                line_list5->points[2*j_temp+1].z = line_list2->points[2*i+1].z;
                line_list5->points[2*j_temp+1].intensity = 1;
            }
            else
            {
                ++j_temp;
                line_list5->points[2*j_temp].x = line_list2->points[2*i].x;
                line_list5->points[2*j_temp].y = line_list2->points[2*i].y;
                line_list5->points[2*j_temp].z = line_list2->points[2*i].z;
                line_list5->points[2*j_temp].intensity = 1;
                line_list5->points[2*j_temp+1].x = line_list2->points[2*i+1].x;
                line_list5->points[2*j_temp+1].y = line_list2->points[2*i+1].y;
                line_list5->points[2*j_temp+1].z = line_list2->points[2*i+1].z;
                line_list5->points[2*j_temp+1].intensity = 1;
                b_left_found = true;
            }
        }
        // back
        if ( alpha_1 >= delta_graph_[4] &&  alpha_1 <= delta_graph_[5]
             && alpha_2 >= delta_graph_[4] &&  alpha_2 <= delta_graph_[5]
             && std::abs(v_cross.norm()) < threshold_similar_line )
        {
            if ( b_back_found )
            {
                line_list5->points[2*j_temp+1].x = line_list2->points[2*i+1].x;
                line_list5->points[2*j_temp+1].y = line_list2->points[2*i+1].y;
                line_list5->points[2*j_temp+1].z = line_list2->points[2*i+1].z;
                line_list5->points[2*j_temp+1].intensity = 2;
            }
            else
            {
                ++j_temp;
                line_list5->points[2*j_temp].x = line_list2->points[2*i].x;
                line_list5->points[2*j_temp].y = line_list2->points[2*i].y;
                line_list5->points[2*j_temp].z = line_list2->points[2*i].z;
                line_list5->points[2*j_temp].intensity = 2;
                line_list5->points[2*j_temp+1].x = line_list2->points[2*i+1].x;
                line_list5->points[2*j_temp+1].y = line_list2->points[2*i+1].y;
                line_list5->points[2*j_temp+1].z = line_list2->points[2*i+1].z;
                line_list5->points[2*j_temp+1].intensity = 2;
                b_back_found = true;
            }
        }
        // right
        if ( alpha_1 >= delta_graph_[6] &&  alpha_1 <= delta_graph_[7]
             && alpha_2 >= delta_graph_[6] &&  alpha_2 <= delta_graph_[7]
             && std::abs(ori_front.dot(v)) < threshold_similar_line )
        {
            if ( b_right_found )
            {
                line_list5->points[2*j_temp+1].x = line_list2->points[2*i+1].x;
                line_list5->points[2*j_temp+1].y = line_list2->points[2*i+1].y;
                line_list5->points[2*j_temp+1].z = line_list2->points[2*i+1].z;
                line_list5->points[2*j_temp+1].intensity = 3;
            }
            else
            {
                ++j_temp;
                line_list5->points[2*j_temp].x = line_list2->points[2*i].x;
                line_list5->points[2*j_temp].y = line_list2->points[2*i].y;
                line_list5->points[2*j_temp].z = line_list2->points[2*i].z;
                line_list5->points[2*j_temp].intensity = 3;
                line_list5->points[2*j_temp+1].x = line_list2->points[2*i+1].x;
                line_list5->points[2*j_temp+1].y = line_list2->points[2*i+1].y;
                line_list5->points[2*j_temp+1].z = line_list2->points[2*i+1].z;
                line_list5->points[2*j_temp+1].intensity = 3;
                b_right_found = true;
            }
        }
    }
    line_list5->resize(2*(j_temp+1));
    // set the message header
    line_list5->header = line_list2->header;
}

/**
 * @brief calculate the center point of the electric tower
 * @param[in]: line_list5
 * @param[in]: b_front_found, b_left_found, b_back_found, b_right_found,
 * @param[out]: center_point
 */
void LaserOdometry::geometryFeatureDetectionSimple(const pcl::PointCloud<pcl::PointXYZI>::Ptr& line_list5,
                                    const bool& b_front_found,
                                    const bool& b_left_found,
                                    const bool& b_back_found,
                                    const bool& b_right_found,
                                    Eigen::Vector4f& center_point)
{
    center_point[3] = 0;
//    if ( b_front_found && b_left_found && b_back_found && b_right_found )
    if ( b_front_found && b_left_found && b_right_found )
    {
        // calcluate dis left and dis right
        Eigen::Vector3f front_left, front_right, left_back, right_back;
        front_left << line_list5->points[1].x, line_list5->points[1].y, line_list5->points[1].z;
        front_right << line_list5->points[0].x, line_list5->points[0].y, line_list5->points[0].z;
        left_back << line_list5->points[3].x, line_list5->points[3].y, line_list5->points[3].z;
        if (b_back_found)
        {
            assert( 8 == line_list5->points.size() );
            right_back << line_list5->points[6].x, line_list5->points[6].y, line_list5->points[6].z;
        }
        else
        {
            assert( 6 == line_list5->points.size() );
            right_back << line_list5->points[4].x, line_list5->points[4].y, line_list5->points[4].z;
        }
        float dis_left = pcl::geometry::distance(front_left, left_back);
        float dis_right = pcl::geometry::distance(front_right, right_back);

        if (dis_left > dis_right)
        {
            center_point.head(3) = (front_right + left_back)/2;
        }
        else
        {
            center_point.head(3) = (front_left + right_back)/2;
        }
        center_point[3] = 1;
    }
    else if (b_front_found && b_left_found && !b_right_found)
    {
        assert( line_list5->points.size() >= 4 );
        Eigen::Vector3f front_right, left_back;
        front_right << line_list5->points[0].x, line_list5->points[0].y, line_list5->points[0].z;
        left_back << line_list5->points[3].x, line_list5->points[3].y, line_list5->points[3].z;
        center_point.head(3) = (front_right + left_back)/2;
        center_point[3] = 1;
    }
    else if (b_front_found && b_right_found && !b_left_found)
    {
        assert( line_list5->points.size() >= 4 );
        Eigen::Vector3f front_left, right_back;
        front_left << line_list5->points[1].x, line_list5->points[1].y, line_list5->points[1].z;
        if ( 4 == line_list5->points.size() )
            right_back << line_list5->points[2].x, line_list5->points[2].y, line_list5->points[2].z;
        else if ( 6 == line_list5->points.size() )
            right_back << line_list5->points[4].x, line_list5->points[4].y, line_list5->points[4].z;
        center_point.head(3) = (front_left + right_back)/2;
        center_point[3] = 1;
    }
}

void LaserOdometry::geometryFeatureDetection(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list4,
        std::vector<int>& graph_indices,
        std::vector<std::vector<int> >& cluster_indices,
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr >& cluster_inliers_clouds,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster_inliers,   // for visualize
        Eigen::Vector3f& front_left,
        Eigen::Vector3f& front_right,
        Eigen::Vector3f& left_back,
        Eigen::Vector3f& right_back,
        Eigen::Vector3f& center_point,
        bool& b_front_end,
        bool& b_left_back,
        bool& b_right_back,
        bool& b_center_point)
{

    ROS_INFO_STREAM("********************* " << line_list4->points.size());

    ROS_INFO_STREAM("--------------------- " << graph_indices.size());
    /** 2-5
     * @brief Refine the center point of the tower
     * through refining clustering the scan_cloud_in to identified line segments in line_list4.
     * Center point calculating through left back ponit or right back point
     * Refine the boundary cloud through clustering scan in to different line segments using Euclidean distance.
     * cluster_indices2 refine each cluster using RANSAC on line model
     * and get the two end points of each line segment
     * Output: cluster_indices - preliminary clustered clouds
     */

    /** 2-5-1
     * @brief Preliminary clustered cloud using an Euclidean distance with respect to the convex boundary line_list4
     * Output: cluster_indices, label each point in scan_cloud_in to identified line segment
     */
    float threshold_p2l = 0.1;
    cluster_indices.resize(line_list4->size()/2);
    /** Whether this filtering step can be merged in the following clustering step? */
    for ( size_t i = 0; i < scan_cloud_in->points.size(); i++ )
    {
        for ( size_t j =0; j < line_list4->size()/2; j++ )
        {
            if (pointToLineDistance(line_list4->points[2*j], line_list4->points[2*j+1], scan_cloud_in->points[i]) < threshold_p2l)
            {
                cluster_indices[j].push_back(i);
            }
        }
    }

    /** 2-5-2
     * @brief Refinement: find the inliers of identified clusters
     * re-calculate the geometry features.
     * create a Point Cloud array/vector
     * refering https://books.google.fr/books?id=JWngSz0L_AEC&pg=PA178&lpg=PA178&dq=std::vector%3Cpcl::PointCloud%3Cpcl::PointXYZ%3E::Ptr+%3E&source=bl&ots=AYJNB-HO4m&sig=hC36qInaVxzkS-GQYqHc8WO_DKs&hl=en&sa=X&ved=0ahUKEwiDlOaJ1rfUAhVCVxoKHdgvA3UQ6AEISjAG#v=onepage&q&f=false
     * Note that this step can be merged with the generation of cluster_indices.
     *
     */
    /**
     * Input: scan_cloud_in and cluster_indices from the last step. Note that this process can be merge with last step (do this later).
     * If not merging the two step, one more for loop needed.
     * Output:
     * cluster_clouds: vector of line segment clouds, one cloud per line segment, the cloud form of scan_cloud_in and cluster_indices
     * cluster_clouds_inliers: the indices of inliers with respect to cluster_clouds to generate cluster_inliers_clouds
     * cluster_inliers_clouds: direct the clustered inliers clouds, merge cluster_clouds and cluster_clouds_inliers
     * convention: inliers refer to the inlier point indices, while inliers_clouds refer to the inlier points.
     * cluster_inliers: the inliers of clustered cloud in one cloud for publishing
     */
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > cluster_clouds;
    std::vector<std::vector<int> > cluster_clouds_inliers(cluster_indices.size());
//    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > cluster_inliers_clouds;
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_inliers(new pcl::PointCloud<pcl::PointXYZ>);    // for visualize
    for (int i = 0; i < cluster_indices.size(); i++)
    {
        // initialize cluster of line segments points, generating cluster_clouds
        // for each line segment
        pcl::PointCloud<pcl::PointXYZ>::Ptr
                cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>(cluster_indices[i].size(), 1));
        for (int j = 0; j < cluster_indices[i].size(); j++)
        {
            cluster_cloud->points[j].x = scan_cloud_in->points[cluster_indices[i][j]].x;
            cluster_cloud->points[j].y = scan_cloud_in->points[cluster_indices[i][j]].y;
            cluster_cloud->points[j].z = scan_cloud_in->points[cluster_indices[i][j]].z;
        }
        cluster_clouds.push_back(cluster_cloud);

        // line fitting, calculate cluster_cloud_inliers
        pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr
                model_l(new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cluster_clouds[i]));
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_l);
        ransac.setDistanceThreshold (.05);
        ransac.computeModel();
        ransac.getInliers(cluster_clouds_inliers[i]);

        // copies all inliers of the model computed to another PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr
                cluster_inlier_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud<pcl::PointXYZ>(*cluster_clouds[i], cluster_clouds_inliers[i], *cluster_inlier_cloud);
        cluster_inliers_clouds.push_back(cluster_inlier_cloud);
        if ( cluster_inlier_cloud->points.size() > 2 )
            *cluster_inliers += *cluster_inlier_cloud;
        else
            graph_indices[i] = -1;
    }
    // set the message header
//    cluster_inliers->header = scan_cloud_in->header;
//    pcl_conversions::toPCL(scan_in->header, cluster_inliers->header);

    /**
     * @brief find the two end points of each line segment
     * Output the tower center point center_point_refined,
     * the refinement of center_point
     * the first step and the second step can also merged using switch logic as // for visualization line_list4 identified version
     */
    // first: fine the two end points of the front line.
    // Note that the two end points are the two farest points on the line.
    // Therefore we get the two end ponits by order the distance between two points
    front_left.setZero();
    front_right.setZero();
    // for visualize the two front end points
    b_front_end = false;
    if ( cluster_inliers_clouds.size() > 0 )
    {
        if (cluster_inliers_clouds[0]->points.size() > 2)
        {
            b_front_end = true;
            // initial the two end points of the front line segment
            front_left[0] = cluster_inliers_clouds[0]->points[0].x;
            front_left[1] = cluster_inliers_clouds[0]->points[0].y;
            front_left[2] = cluster_inliers_clouds[0]->points[0].z;
            front_right[0] = cluster_inliers_clouds[0]->points[1].x;
            front_right[1] = cluster_inliers_clouds[0]->points[1].y;
            front_right[2] = cluster_inliers_clouds[0]->points[1].z;
            float dis = (front_left-front_right).norm();
            float dis1 = 0;
            float dis2 = 0;

            for (int i = 2; i < cluster_inliers_clouds[0]->points.size(); i++)
            {
                Eigen::Vector3f pt(cluster_inliers_clouds[0]->points[i].x,
                                    cluster_inliers_clouds[0]->points[i].y,
                                    cluster_inliers_clouds[0]->points[i].z);
                dis1 = (front_left - pt).norm();
                dis2 = (front_right - pt).norm();
                // Due to geometry relationship,
                // if there is a point outside the current two point segment, update
                if (dis1 > dis || dis2 > dis)
                {
                    if (dis1 > dis2)
                    {
                        front_right = pt;
                        dis = dis1;
                    }
                    else
                    {
                        front_left = pt;
                        dis = dis2;
                    }
                }
            }
            // decide the left and the right
            if (std::atan2(front_left[1], front_left[0]) < std::atan2(front_right[1], front_right[0]))
            {
                Eigen::Vector3f pt = front_left;
                front_left = front_right;
                front_right = pt;
            }
        }
    }

    // Second: find the furthest left back or right back points
    // this part is similar with // for visualization line_list4 identified version
    // this part is similar with rectangle fitting from some point of view
    b_left_back = false;
    b_right_back = false;
    float dis_left_back = 0;
    float dis_right_back = 0;
    center_point.setZero() ;
    left_back.setZero();
    right_back.setZero();
    if (cluster_inliers_clouds.size() == graph_indices.size())
    {
        if (cluster_inliers_clouds.size() > 1 && b_front_end)
        {
            for (int i = 1; i < cluster_inliers_clouds.size(); i++)
            {
                switch (graph_indices[i])
                {
                    // left
                    case 1:
                    for (int j = 0; j < cluster_inliers_clouds[i]->points.size(); j++)
                    {
                        Eigen::Vector3f pt(cluster_inliers_clouds[i]->points[j].x,
                                            cluster_inliers_clouds[i]->points[j].y,
                                            cluster_inliers_clouds[i]->points[j].z);
                        float dis = (front_left - pt).norm();
                        if (dis_left_back < dis)
                        {
                            left_back = pt;
                            dis_left_back = dis;
                        }
                    }
                    b_left_back = true;
                    break;
                    // right
                    case 3:
                    for (int j = 0; j < cluster_inliers_clouds[i]->points.size(); j++)
                    {
                        Eigen::Vector3f pt(cluster_inliers_clouds[i]->points[j].x,
                                            cluster_inliers_clouds[i]->points[j].y,
                                            cluster_inliers_clouds[i]->points[j].z);
                        float dis = (front_right - pt).norm();
                        if (dis_right_back < dis)
                        {
                            right_back = pt;
                            dis_right_back = dis;
                        }
                    }
                    b_right_back = true;
                    break;
                }
            }
        }
    }
    else
       ROS_WARN_STREAM("The size of cluster_inliers_clouds is not equal to graph_indices ...");

    /** Third
     * @brief the center_point of the tower
     */
    b_center_point = false;
    if (b_left_back && b_right_back)
    {
        if (dis_left_back > dis_right_back)
        {
            center_point = (front_right + left_back)/2;
        }
        else
        {
            center_point = (front_left + right_back)/2;
        }
        b_center_point = true;
    }
    else if (b_left_back)
    {
        center_point = (front_right + left_back)/2;
        b_center_point = true;
    }
    else if (b_right_back)
    {
        center_point = (front_left + right_back)/2;
        b_center_point = true;
    }
}

/**
 * @brief geometry feature detection through rectangle fitting in laser frame
 *
 * @param[in]: scan_cloud_in, line_list5, b_front_found, b_left_found, b_back_found, b_right_found
 * @param[out]: geometry points and their states in laser frame
 * @param[out]: four cloud for four boundary lines in laser frame
 */
void LaserOdometry::geometryFeatureDetection2(
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in,
            const pcl::PointCloud<pcl::PointXYZI>::Ptr& line_list5,
            const bool& b_front_found,
            const bool& b_left_found,
            const bool& b_back_found,
            const bool& b_right_found,
            Eigen::Vector4f& pt_front_left,
            Eigen::Vector4f& pt_front_right,
            Eigen::Vector4f& pt_front_middle,
            Eigen::Vector4f& pt_left_back,
            Eigen::Vector4f& pt_right_back,
            Eigen::Vector4f& pt_center,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_front_line,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_left_line,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_back_line,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_right_line)
{
    assert( 8 == line_list5->size() );  // in case of complete boundary observed

    // clustering scan in
    float threshold_p2l = 0.05;
    for ( size_t i = 0; i < scan_cloud_in->points.size(); ++i )
    {
        for ( size_t j = 0; j < line_list5->size()/2; ++j )
        {
            pcl::PointXYZ pt1, pt2;
            pt1.x = line_list5->points[2*j].x;
            pt1.y = line_list5->points[2*j].y;
            pt1.z = line_list5->points[2*j].z;
            pt2.x = line_list5->points[2*j+1].x;
            pt2.y = line_list5->points[2*j+1].y;
            pt2.z = line_list5->points[2*j+1].z;
            if (pointToLineDistance(pt1, pt2, scan_cloud_in->points[i]) < threshold_p2l)
            {
                if ( std::abs(line_list5->points[2*j].intensity - 0) < 0.01 )
                    cloud_front_line->points.push_back(scan_cloud_in->points[i]);
                if ( std::abs(line_list5->points[2*j].intensity - 1) < 0.01 )
                    cloud_left_line->points.push_back(scan_cloud_in->points[i]);
                if ( std::abs(line_list5->points[2*j].intensity - 2) < 0.01 )
                    cloud_back_line->points.push_back(scan_cloud_in->points[i]);
                if ( std::abs(line_list5->points[2*j].intensity - 3) < 0.01 )
                    cloud_right_line->points.push_back(scan_cloud_in->points[i]);
            }
        }
    }

    	// line fitting using rectangle model
	// front line: c0 + a0*x + b0*y = 0
	// left line: c1 - b0*x + a0*y = 0
	// back line: c2 + a0*x + b0*y = 0
	// right line: c3 - b0*x + a0*y = 0
	// linear least squares problem type Ax = 0 subject to a0^2+b0^2 = 1
	// x = [ c0 a0 b0 ]^{T}
	int rows_A = cloud_front_line->points.size() + cloud_left_line->points.size()
			+ cloud_back_line->points.size() + cloud_right_line->points.size();
	int cols_A = 6;
	Eigen::MatrixXf A(rows_A, cols_A); 
	A.setZero();
	Eigen::VectorXf x(cols_A);
	x.setZero();

	// set A
	// front line
	for ( size_t i = 0; i < cloud_front_line->points.size(); ++i )
	{
		A(i, 0) = 1;
		A(i, 1) = 0;
		A(i, 2) = 0;
		A(i, 3) = 0;
		A(i, 4) = cloud_front_line->points[i].x;
		A(i, 5) = cloud_front_line->points[i].y;
	}
	// left line
	for ( size_t i = 0; i < cloud_left_line->points.size(); ++i )
	{
		int j = cloud_front_line->points.size() + i;
		A(j, 0) = 0;
		A(j, 1) = 1;
		A(j, 2) = 0;
		A(j, 3) = 0;
		A(j, 4) = cloud_left_line->points[i].y;
		A(j, 5) = -cloud_left_line->points[i].x;
	}
	// back line
	for ( size_t i = 0; i < cloud_back_line->points.size(); ++i )
	{
		int j = cloud_front_line->points.size() + cloud_left_line->points.size() + i;
		A(j, 0) = 0;
		A(j, 1) = 0;
		A(j, 2) = 1;
		A(j, 3) = 0;
		A(j, 4) = cloud_back_line->points[i].x;
		A(j, 5) = cloud_back_line->points[i].y;
	}
	// right line
	for ( size_t i = 0; i < cloud_right_line->points.size(); ++i )
	{
		int j = cloud_front_line->points.size() + cloud_left_line->points.size() 
			+ cloud_back_line->points.size() + i;
		A(j, 0) = 0;
		A(j, 1) = 0;
		A(j, 2) = 0;
		A(j, 3) = 1;
		A(j, 4) = cloud_right_line->points[i].y;
		A(j, 5) = -cloud_right_line->points[i].x;
	}

	// solve the least-squares problem using SVD
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A.transpose()*A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	int rows_V = svd.matrixV().rows();
	int cols_V = svd.matrixV().cols();
	assert( cols_A == rows_V && cols_A == cols_V );
	Eigen::MatrixXf matrixV(rows_V, cols_V);
	matrixV = svd.matrixV();
	x = matrixV.col(cols_V-1);
	
	// calculate the geometry relationship ( points of the rectangle )
	Eigen::VectorXf b_pt(2);
	b_pt.setZero();
	Eigen::VectorXf x_pt(2);
	x_pt.setZero();
	// front left point
	Eigen::MatrixXf A_pt(2, 2); 
	A_pt << x(4), x(5),
		-x(5), x(4);
	b_pt << -x(0), -x(1);
	x_pt = A_pt.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_pt);
	pt_front_left << x_pt(0), x_pt(1), 0, 1;

	// front right point
	A_pt << x(4), x(5),
		-x(5), x(4);
	b_pt << -x(0), -x(3);
	x_pt = A_pt.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_pt);
	pt_front_right << x_pt(0), x_pt(1), 0, 1;

	// left back
	A_pt << x(4), x(5),
		-x(5), x(4);
	b_pt << -x(2), -x(1);
	x_pt = A_pt.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_pt);
	pt_left_back << x_pt(0), x_pt(1), 0 , 1;

	// right back
	A_pt << x(4), x(5),
		-x(5), x(4);
	b_pt << -x(2), -x(3);
	x_pt = A_pt.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_pt);
	pt_right_back << x_pt(0), x_pt(1), 0 , 1;

	// center points
	pt_center << (pt_front_left.head(2) + pt_front_right.head(2) + pt_left_back.head(2) + pt_right_back.head(2))/4, 0 ,1;
}

/**
 * @brief Visualize the geometry feature points of the object
 * two end ponits of the front line segment (left blue right yellow)
 * the left back (red) and the right back (green)
 * tower center red
 * @param[out]: object_feature_points
 */
void LaserOdometry::geometryFeatureVisualization(const Eigen::Vector4f& front_left,
                                  const Eigen::Vector4f& front_right,
                                  const Eigen::Vector4f& left_back,
                                  const Eigen::Vector4f& right_back,
                                  const Eigen::Vector4f& center_point,
			        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_front_line,
				const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_left_line,
				const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_back_line,
				const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_right_line,
                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr& object_feature_points,
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_clustered)
{
    // label the front right yellow
    if ( std::abs( front_right[3] -1 ) < 0.01 )
    {
        pcl::PointXYZRGB p;
        p.x = front_right[0];
        p.y = front_right[1];
        p.z = front_right[2];
        p.r = 255;
        p.g = 255;
        p.b = 0;
        object_feature_points->push_back(p);
    }
    // label the front left blue
    if ( std::abs( front_left[3] -1 ) < 0.01 )
    {
        pcl::PointXYZRGB p;
        p.x = front_left[0];
        p.y = front_left[1];
        p.z = front_left[2];
        p.r = 0;
        p.g = 0;
        p.b = 255;
        object_feature_points->push_back(p);
    }
    // label the left back red
    if ( std::abs( left_back[3] - 1 ) < 0.01 )
    {
        pcl::PointXYZRGB p;
        p.x = left_back[0];
        p.y = left_back[1];
        p.z = left_back[2];
        p.r = 255;
        p.g = 0;
        p.b = 0;
        object_feature_points->push_back(p);
    }
    // label the right back green
    if ( std::abs( right_back[3] - 1 ) < 0.01 )
    {
        pcl::PointXYZRGB p;
        p.x = right_back[0];
        p.y = right_back[1];
        p.z = right_back[2];
        p.r = 0;
        p.g = 255;
        p.b = 0;
        object_feature_points->push_back(p);
    }
    // label the center point red
    if ( std::abs( center_point[3] - 1 ) < 0.01 )
    {
        pcl::PointXYZRGB p;
        p.x = center_point[0];
        p.y = center_point[1];
        p.z = center_point[2];
        p.r = 255;
        p.g = 0;
        p.b = 0;
        object_feature_points->push_back(p);
    }

	// front line - red
	for (size_t i = 0; i < cloud_front_line->points.size(); ++i)
	{
		pcl::PointXYZRGB p;
                p.x = cloud_front_line->points[i].x;
                p.y = cloud_front_line->points[i].y;
                p.z = cloud_front_line->points[i].z + 0.2;
		p.r = 255;
                p.g = 0;
                p.b = 0;
		cloud_clustered->push_back(p);
	}
	// left line - green
	for (size_t i = 0; i < cloud_left_line->points.size(); ++i)
	{
		pcl::PointXYZRGB p;
                p.x = cloud_left_line->points[i].x;
                p.y = cloud_left_line->points[i].y;
                p.z = cloud_left_line->points[i].z + 0.4;
		p.r = 0;
                p.g = 255;
                p.b = 0;
		cloud_clustered->push_back(p);
	}
	// back line - blue
	for (size_t i = 0; i < cloud_back_line->points.size(); ++i)
	{
		pcl::PointXYZRGB p;
                p.x = cloud_back_line->points[i].x;
                p.y = cloud_back_line->points[i].y;
                p.z = cloud_back_line->points[i].z + 0.6;
		p.r = 0;
                p.g = 0;
                p.b = 255;
		cloud_clustered->push_back(p);
	}
	// right line - orange
	for (size_t i = 0; i < cloud_right_line->points.size(); ++i)
	{
		pcl::PointXYZRGB p;
                p.x = cloud_right_line->points[i].x;
                p.y = cloud_right_line->points[i].y;
                p.z = cloud_right_line->points[i].z + 0.8;
		p.r = 255;
                p.g = 126;
                p.b = 0;
		cloud_clustered->push_back(p);
	}
}

/**
 * @brief Geometry Feature based Pose Estimation:
 * (1) roll and pitch is from the gamma of imu data directly.
 * (2) If the front line detected, yaw is decidedm, then the rotation is obtained.
 * (3) z is decided from the laser altitude and the rotation.
 * (4) If the center point is detected, x and y are decided, then the translation is obtained.
 *
 * @param[out]: w_R_v, rotation of the vehicle (laser frame) with respect to the world frame
 * (NWU, ros world frame)
 * @param[out]: w_t_v, translation of the vehicle (laser frame) with respect to the world frame of the front line
 */
void LaserOdometry::geometryPoseEstimation_frontLine(const Eigen::Vector3d& imu_gamma,
                                           const double& laser_altimeter,
                                           const Eigen::Vector4f& pt_front_left,
                                           const Eigen::Vector4f& pt_front_right,
				           Eigen::Vector3f& i_rot_imu,
                                           Eigen::Vector3f& w_t_v,
					   Eigen::Matrix3f& laser_R_object_fl,
                                           Eigen::Vector3f& laser_t_object_fl)
{
    // initialize the output
    i_rot_imu << 0, 0, 0;
    w_t_v << 0, 0, 0;
    laser_R_object_fl.setZero();
    laser_t_object_fl << 0, 0, 0;
    
    // roll and pitch from imu gamma
    double pitch = std::asin(-imu_gamma[0]);
    double roll = std::atan2(imu_gamma[1], imu_gamma[2]);

    // yaw from front line segment detected by laser scanner
    double yaw = 0;
    Eigen::Vector4f pt_front_center(0, 0, 0, 0);
    if ( std::abs(pt_front_left[3]-1) < 0.01 && std::abs(pt_front_right[3]-1) < 0.01 )
    {
        Eigen::Vector3f v = pt_front_left.head(3) - pt_front_right.head(3);
        // object frame (global fixed) with respect to laser frame
        double theta = std::atan2(v[1], v[0]) - M_PI/2;   // x-axis direction from y-axis direction
        // theta is the object frame with respect to laser frame,
        // -theta is laser frame with respect to ros world frame,
        // theta is also imu with respect to the defined inertial world frame.
        yaw = theta;

        // front center
        pt_front_center.head(3) = pt_front_left.head(3) + pt_front_right.head(3);
        pt_front_center.head(3) /= 2;
	pt_front_center(3) = 1;

	laser_R_object_fl = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
	laser_t_object_fl << pt_front_center.head(3);
    }
    // set the rotation vector
    i_rot_imu << roll, pitch, yaw;

    // calculate the rotation matrix
    Eigen::Matrix3f Rx, Ry, Rz, R_x_pi, i_R_imu, w_R_v;
    R_x_pi = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX());
    Rx = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
    Ry = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
    Rz = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
    i_R_imu = Rz * Ry * Rx;
    w_R_v = R_x_pi*i_R_imu*R_x_pi;

    // calculate altitude w_z_v
    // stand at the projection point observe the UAV alining UAV attitude.
    float w_z_v = 0;
    Eigen::Vector3f laser_z(0, 0, laser_altimeter);
    Eigen::Vector3f wROS_z(0, 0, 0);
    wROS_z = w_R_v*laser_z;
    w_z_v = wROS_z[2];

    // calculate tranlation components w_x_v and w_y_v with respect to the center point of the front line
    float w_x_v = 0;
    float w_y_v = 0;
    Eigen::Matrix3f laser_R_object;
    if ( std::abs(pt_front_center[3]-1) < 0.01 )
    {
        // obtain rotation matrix of the laser frame with respect to tower frame, also ros world frame
        // in planer
        // recall that yaw is also the object frame with respect to laser frame
        laser_R_object = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
        // using the middle point of the geometry center of the tower.
        // wROS_t_laser: is the translation of laser frame with respect to object frame in 2D
        // they are also laser frame with respect to the ros world frame
        Eigen::Vector3f wROS_t_laser = - laser_R_object.transpose()*pt_front_center.head(3);
        w_x_v = wROS_t_laser[0];
        w_y_v = wROS_t_laser[1];

    	// set translation vector
    	w_t_v << w_x_v, w_y_v, w_z_v;
    }
}

/**
 * @brief Geometry Feature based Pose Estimation:
 * (1) roll and pitch is from the gamma of imu data directly.
 * (2) If the front line detected, yaw is decidedm, then the rotation is obtained.
 * (3) z is decided from the laser altitude and the rotation.
 * (4) If the square (or the center point) is detected, x and y are decided, then the translation is obtained.
 *
 * @param[in]: imu_gamma, altimeter laser, geometry feature points
 * @param[out]: w_R_v - rotation matrix of the vehicle (laser frame) with respect to the world frame
 * (NWU, ros world frame)
 * @param[out]: i_rot_imu - rotation vector of the imu frame (local NED) with respect to the inertial frame (NED), not ros world 
 * @param[out]: w_t_v - translation vector of the vehicle (or laser) frame with respect to the world frame (NWU)
 */
void LaserOdometry::geometryPoseEstimation(const Eigen::Vector3d& imu_gamma,
                                           const double& laser_altimeter,
                                           const Eigen::Vector4f& pt_front_left,
                                           const Eigen::Vector4f& pt_front_right,
                                           const Eigen::Vector4f& pt_center,
                                           Eigen::Matrix3f& w_R_v,
				           Eigen::Vector3f& i_rot_imu,
                                           Eigen::Vector3f& w_t_v,
                                           Eigen::Matrix3f& laser_R_object,
                                           Eigen::Vector3f& laser_t_object,
					   Eigen::Matrix3f& laser_R_object_fl,
                                           Eigen::Vector3f& laser_t_object_fl,
                                           bool& b_rotation_ok,
                                           bool& b_translation_ok)
{
    // roll and pitch from imu gamma
    double pitch = std::asin(-imu_gamma[0]);
    double roll = std::atan2(imu_gamma[1], imu_gamma[2]);

    // yaw from front line segment detected by laser scanner
    double yaw = 0;
    if ( std::abs(pt_front_left[3]-1) < 0.01 && std::abs(pt_front_right[3]-1) < 0.01 )
    {
        Eigen::Vector3f v = pt_front_left.head(3) - pt_front_right.head(3);
        // object frame (global fixed) with respect to laser frame
        double theta = std::atan2(v[1], v[0]) - M_PI/2;   // x-axis direction from y-axis direction
        // theta is the object frame with respect to laser frame,
        // -theta is laser frame with respect to ros world frame,
        // theta is also imu with respect to the defined inertial world frame.
        yaw = theta;
        b_rotation_ok = true;
	laser_t_object_fl << pt_front_left.head(3) + pt_front_right.head(3);
	laser_t_object_fl /= 2;
	laser_R_object_fl = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
    }

    // calculate the rotation
    Eigen::Matrix3f Rx, Ry, Rz, R_x_pi, i_R_imu;
    R_x_pi = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX());
    Rx = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
    Ry = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
    Rz = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
    i_R_imu = Rz * Ry * Rx;
    w_R_v = R_x_pi*i_R_imu*R_x_pi;
    i_rot_imu << roll, pitch, yaw;

    // calculate altitude w_z_v
    // stand at the projection point observe the UAV alining UAV attitude.
    double w_z_v = 0;
    Eigen::Vector3f laser_z(0, 0, laser_altimeter);
    Eigen::Vector3f wROS_z(0, 0, 0);
    wROS_z = w_R_v*laser_z;
    w_z_v = wROS_z[2];

    // calculate tranlation components w_x_v and w_y_v
    double w_x_v = 0;
    double w_y_v = 0;
    if ( std::abs(pt_center[3]-1) < 0.01 )
    {
        // obtain rotation matrix of the laser frame with respect to tower frame, also ros world frame
        // in planer
        // recall that yaw is also the object frame with respect to laser frame
        laser_R_object = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
        // using the middle point of the geometry center of the tower.
        // wROS_t_laser: is the translation of laser frame with respect to object frame in 2D
        // they are also laser frame with respect to the ros world frame
        Eigen::Vector3f wROS_t_laser = - laser_R_object.transpose()*pt_center.head(3);
        w_x_v = wROS_t_laser[0];
        w_y_v = wROS_t_laser[1];

	laser_t_object << pt_center.head(3);
        b_translation_ok = true;
    }

    // set translation vector
    w_t_v << w_x_v, w_y_v, w_z_v;
}

/**
* \brief register the current laser scan to world frame (ROS world frame)
*
* \param[in] scan_cloud_in (laser frame), w_R_v, w_t_v, pt_center (laser frame) 
* \param[in] cloud_front_line, cloud_left_line, cloud_back_line, cloud_right_line in laser frame
* \param[out] cloud_worldROS_all, all the cloud in the ROS world frame
* \param[out] cloud_worldROS_geometry, cloud from geometry method in the ROS world frame
* \param[out] cloud_worldROS_fail, cloud when geometry method fail in the ROS world frame
* \param[out] cloud_front_plane, cloud_left_plane, cloud_back_plane, cloud_right_plane in the ROS world frame
*/
void LaserOdometry::cloudRegistration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in,
		   const Eigen::Matrix3f& w_R_v, const Eigen::Vector3f& w_t_v,
		   const Eigen::Vector4f& pt_center, const std::string& worldROS_frame,
		   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_front_line,
		   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_left_line,
		   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_back_line,
		   const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_right_line,
		   pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_worldROS_all,
		   pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_worldROS,
		   pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_worldROS_fail,
		   std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_front_plane,
		   std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_left_plane,
		   std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_back_plane,
		   std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_right_plane)
{
    // define a transform
    Eigen::Affine3f wROS_T_laser = Eigen::Affine3f::Identity();
    // set a translation
    wROS_T_laser.translation() = w_t_v;
    // set the rotation matrix
    wROS_T_laser.rotate(w_R_v);
    pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>(scan_cloud_in->points.size(), 1));
    // apply transformation from world ROS (or defined inertial) to laser
    pcl::transformPointCloud(*scan_cloud_in, *scan_cloud_transformed, wROS_T_laser);
    *cloud_worldROS_all += *scan_cloud_transformed;
    // set the message header
    cloud_worldROS_all->header.frame_id = worldROS_frame;
    cloud_worldROS_all->header.stamp = scan_cloud_in->header.stamp;
    if ( std::abs( pt_center[3] - 1 ) < 0.01 )
    {
        *cloud_worldROS += *scan_cloud_transformed;
        // set the message header
        cloud_worldROS->header.frame_id = worldROS_frame;
	cloud_worldROS->header.stamp = scan_cloud_in->header.stamp;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_front_transformed(new pcl::PointCloud<pcl::PointXYZ>(cloud_front_line->points.size(), 1));
	pcl::transformPointCloud(*cloud_front_line, *cloud_front_transformed, wROS_T_laser);
        cloud_front_plane.push_back(cloud_front_transformed);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_left_transformed(new pcl::PointCloud<pcl::PointXYZ>(cloud_left_line->points.size(), 1));
	pcl::transformPointCloud(*cloud_left_line, *cloud_left_transformed, wROS_T_laser);
        cloud_left_plane.push_back(cloud_left_transformed);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_back_transformed(new pcl::PointCloud<pcl::PointXYZ>(cloud_back_line->points.size(), 1));
	pcl::transformPointCloud(*cloud_back_line, *cloud_back_transformed, wROS_T_laser);
        cloud_back_plane.push_back(cloud_back_transformed);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_right_transformed(new pcl::PointCloud<pcl::PointXYZ>(cloud_right_line->points.size(), 1));
	pcl::transformPointCloud(*cloud_right_line, *cloud_right_transformed, wROS_T_laser);
        cloud_right_plane.push_back(cloud_right_transformed);
    }
    else
    {
        *cloud_worldROS_fail += *scan_cloud_transformed;
        // set the message header
        cloud_worldROS_fail->header.frame_id = worldROS_frame;
	cloud_worldROS_fail->header.stamp = scan_cloud_in->header.stamp;
    }
}

void LaserOdometry::cloudPlaneVisualization(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_front_plane,
				   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_left_plane,
				   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_back_plane,
				   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_right_plane,
				   pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_planes)
{
	pcl::PointXYZRGB pt;
	// front plane
        pt.r = 255;
        pt.g = 255;
        pt.b = 255;
	for (int i = 0; i < cloud_front_plane.size(); ++i)
	{
		for (int j = 0; j < cloud_front_plane[i]->points.size(); ++j)
		{
			pt.x = cloud_front_plane[i]->points[j].x;
			pt.y = cloud_front_plane[i]->points[j].y;
			pt.z = cloud_front_plane[i]->points[j].z;
			cloud_planes->push_back(pt);
		}
	}

	// left plane
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
	for (int i = 0; i < cloud_left_plane.size(); ++i)
	{
		for (int j = 0; j < cloud_left_plane[i]->points.size(); ++j)
		{
			pt.x = cloud_left_plane[i]->points[j].x;
			pt.y = cloud_left_plane[i]->points[j].y;
			pt.z = cloud_left_plane[i]->points[j].z;
			cloud_planes->push_back(pt);
		}
	}

	// back plane
        pt.r = 255;
        pt.g = 255;
        pt.b = 255;
	for (int i = 0; i < cloud_back_plane.size(); ++i)
	{
		for (int j = 0; j < cloud_back_plane[i]->points.size(); ++j)
		{
			pt.x = cloud_back_plane[i]->points[j].x;
			pt.y = cloud_back_plane[i]->points[j].y;
			pt.z = cloud_back_plane[i]->points[j].z;
			cloud_planes->push_back(pt);
		}
	}

	// right plane
        pt.r = 0;
        pt.g = 0;
        pt.b = 255;
	for (int i = 0; i < cloud_right_plane.size(); ++i)
	{
		for (int j = 0; j < cloud_right_plane[i]->points.size(); ++j)
		{
			pt.x = cloud_right_plane[i]->points[j].x;
			pt.y = cloud_right_plane[i]->points[j].y;
			pt.z = cloud_right_plane[i]->points[j].z;
			cloud_planes->push_back(pt);
		}
	}
}

/**
 * @brief plane model fiting
 *
 * @param[in]: cloud_front_plane, cloud_left_plane, cloud_back_plane, cloud_right_plane
 * @param[out]: param_front_plane, param_left_plane, param_back_plane, param_right_plane
 */
void LaserOdometry::planeModelFitting(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_front_plane,
		  		      const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_left_plane,
				      const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_back_plane,
				      const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_right_plane,
				      Eigen::Vector4f& param_front_plane, Eigen::Vector4f& param_left_plane,
				      Eigen::Vector4f& param_back_plane, Eigen::Vector4f& param_right_plane)
{
	// plane fitting in Hessian Normal form: p + a*x + b*y + c*z = 0
	// with constraints
	// front plane: P0 + a0*x + b0*y + c0*z = 0
	// left plane: p1 - b0*x + a0*y + c1*z = 0
	// back plane: p2 + a0*x + b0*y - c0*z = 0
	// right plane: p3 - b0*x + a0*y - c1*z = 0
	// linear least squares problem type Ax = 0 subject to a0^2+b0^2 + c0^2 = 1
	// unknows values x = [ p0 p1 p2 p3 a0 b0 c0 c1 ]^{T}
	// front plane
	int length_front = 0;
	for (int i = 0; i < cloud_front_plane.size(); ++i)
		length_front += cloud_front_plane[i]->points.size();
	// left plane
	int length_left = 0;	
	for (int i = 0; i < cloud_left_plane.size(); ++i)
		length_left += cloud_left_plane[i]->points.size();
	// back plane
	int length_back = 0;
	for (int i = 0; i < cloud_back_plane.size(); ++i)
		length_back += cloud_back_plane[i]->points.size();
	// right plane
	int length_right = 0;
	for (int i = 0; i < cloud_right_plane.size(); ++i)
		length_right += cloud_right_plane[i]->points.size();
	int rows_A = length_front + length_left + length_back + length_right;
	int cols_A = 8;
	Eigen::MatrixXf A(rows_A, cols_A); 
	A.setZero();
	Eigen::VectorXf x(cols_A);
	x.setZero();

	// set A
	// front plane
	int k = 0;
	for (int j = 0; j < cloud_front_plane.size(); ++j)
	{
		for ( int i = 0; i < cloud_front_plane[j]->points.size(); ++i )
		{
			A(i, 0) = 1;
			A(i, 1) = 0;
			A(i, 2) = 0;
			A(i, 3) = 0;
			A(i, 4) = cloud_front_plane[j]->points[i].x;
			A(i, 5) = cloud_front_plane[j]->points[i].y;
			A(i, 6) = cloud_front_plane[j]->points[i].z;
			A(i, 7) = 0;
		}
	}
	k += length_front;
	// left plane
	for (int j = 0; j < cloud_left_plane.size(); ++j)
	{
		for ( int i = k; i < k + cloud_left_plane[j]->points.size(); ++i )
		{
			assert(i < rows_A);
			A(i, 0) = 0;
			A(i, 1) = 1;
			A(i, 2) = 0;
			A(i, 3) = 0;
			A(i, 4) = cloud_left_plane[j]->points[i].y;
			A(i, 5) = -cloud_left_plane[j]->points[i].x;
			A(i, 6) = 0;
			A(i, 7) = cloud_left_plane[j]->points[i].z;
		}
	}
	k += length_left;
	// back plane
	for (int j = 0; j < cloud_back_plane.size(); ++j)
	{
		for ( int i = k; i < k + cloud_back_plane[j]->points.size(); ++i )
		{
			assert(i < rows_A);
			A(i, 0) = 0;
			A(i, 1) = 0;
			A(i, 2) = 1;
			A(i, 3) = 0;
			A(i, 4) = cloud_back_plane[j]->points[i].x;
			A(i, 5) = cloud_back_plane[j]->points[i].y;
			A(i, 6) = -cloud_back_plane[j]->points[i].z;
			A(i, 7) = 0;
		}
	}
	k += length_back;
	// right plane
	for (int j = 0; j < cloud_right_plane.size(); ++j)
	{
		for ( size_t i = k; i < k + cloud_right_plane[j]->points.size(); ++i )
		{
			assert(i < rows_A);
			A(i, 0) = 0;
			A(i, 1) = 0;
			A(i, 2) = 0;
			A(i, 3) = 1;
			A(i, 4) = cloud_right_plane[j]->points[i].y;
			A(i, 5) = -cloud_right_plane[j]->points[i].x;
			A(i, 6) = 0;
			A(i, 7) = -cloud_right_plane[j]->points[i].z;
		}
	}

	// solve the least-squares problem using SVD
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A.transpose()*A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	int rows_V = svd.matrixV().rows();
	int cols_V = svd.matrixV().cols();
	assert( cols_A == rows_V && cols_A == cols_V );
	Eigen::MatrixXf matrixV(rows_V, cols_V);
	matrixV = svd.matrixV();
	x = matrixV.col(cols_V-1);

	// set plane parameters
	float d1 = std::sqrt(x(4)*x(4) + x(5)*x(5) + x(6)*x(6)); 
	float d2 = std::sqrt(x(4)*x(4) + x(5)*x(5) + x(7)*x(7));
	param_front_plane << x(4), x(5), x(6), x(0);
	param_left_plane << -x(5), x(4), x(7), x(1);
	param_back_plane << x(4), x(5), -x(6), x(2);
	param_right_plane << -x(5), x(4), -x(7), x(3);
	std::cout << "param front plane: " << param_front_plane << std::endl;
	std::cout << "param left plane: " << param_left_plane << std::endl;
	std::cout << "param back plane: " << param_back_plane << std::endl;
	std::cout << "param right plane: " << param_right_plane << std::endl;
	param_front_plane = param_front_plane/d1;
	param_left_plane = param_left_plane/d2;
	param_back_plane = param_back_plane/d1;
	param_right_plane = param_right_plane/d2;
	std::cout << "normalized front plane: " << param_front_plane << std::endl;
	std::cout << "normalized left plane: " << param_left_plane << std::endl;
	std::cout << "normalized back plane: " << param_back_plane << std::endl;
	std::cout << "normalized right plane: " << param_right_plane << std::endl;
}

/**
 * @brief plane model fiting individually
 *
 * @param[in]: cloud_front_plane, cloud_left_plane, cloud_back_plane, cloud_right_plane
 * @param[out]: param_front_plane, param_left_plane, param_back_plane, param_right_plane
 */
void LaserOdometry::planeModelFittingTest(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_front_plane,
		  		      const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_left_plane,
				      const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_back_plane,
				      const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_right_plane,
				      Eigen::Vector4f& param_front_plane, Eigen::Vector4f& param_left_plane,
				      Eigen::Vector4f& param_back_plane, Eigen::Vector4f& param_right_plane)
{
	// plane fitting in Hessian Normal form: p + a*x + b*y + c*z = 0
	// with constraints
	// front plane: P0 + a0*x + b0*y + c0*z = 0
	// left plane: p1 - b0*x + a0*y + c1*z = 0
	// back plane: p2 + a0*x + b0*y - c0*z = 0
	// right plane: p3 - b0*x + a0*y - c1*z = 0
	// linear least squares problem type Ax = 0 subject to a0^2+b0^2 + c0^2 = 1
	// unknows values x = [ p0 p1 p2 p3 a0 b0 c0 c1 ]^{T}
	int rows_A_front = 0;
	// front plane
	for (int i = 0; i < cloud_front_plane.size(); ++i)
		rows_A_front += cloud_front_plane[i]->points.size();
	int cols_A = 4;
	Eigen::MatrixXf A_front(rows_A_front, cols_A); 
	A_front.setZero();
	Eigen::VectorXf x(cols_A);
	x.setZero();

	// set A
	// front plane
	for (int j = 0; j < cloud_front_plane.size(); ++j)
	{
		for ( size_t i = 0; i < cloud_front_plane[j]->points.size(); ++i )
		{
			A_front(i, 0) = 1;
			A_front(i, 1) = cloud_front_plane[j]->points[i].x;
			A_front(i, 2) = cloud_front_plane[j]->points[i].y;
			A_front(i, 3) = cloud_front_plane[j]->points[i].z;
		}
	}
	// solve the least-squares problem using SVD
	Eigen::JacobiSVD<Eigen::MatrixXf> svd_front(A_front.transpose()*A_front, Eigen::ComputeThinU | Eigen::ComputeThinV);
	int rows_V_front = svd_front.matrixV().rows();
	int cols_V_front = svd_front.matrixV().cols();
	assert( cols_A == rows_V_front && cols_A == cols_V_front );
	Eigen::MatrixXf matrixV_front(rows_V_front, cols_V_front);
	matrixV_front = svd_front.matrixV();
	x = matrixV_front.col(cols_V_front-1);
	// set plane parameters
	param_front_plane << x(1), x(2), x(3), x(0);
	float d = std::sqrt(x(1)*x(1) + x(2)*x(2) + x(3)*x(3));
	param_front_plane = param_front_plane/d;

	// left plane
	int rows_A_left = 0;
	// front plane
	for (int i = 0; i < cloud_left_plane.size(); ++i)
		rows_A_left += cloud_left_plane[i]->points.size();
	Eigen::MatrixXf A_left(rows_A_left, cols_A); 
	A_left.setZero();
	for (int j = 0; j < cloud_left_plane.size(); ++j)
	{
		for ( size_t i = 0; i < cloud_left_plane[j]->points.size(); ++i )
		{
			A_left(i, 0) = 1;
			A_left(i, 1) = cloud_left_plane[j]->points[i].x;
			A_left(i, 2) = cloud_left_plane[j]->points[i].y;
			A_left(i, 3) = cloud_left_plane[j]->points[i].z;
		}
	}
	// solve the least-squares problem using SVD
	Eigen::JacobiSVD<Eigen::MatrixXf> svd_left(A_left.transpose()*A_left, Eigen::ComputeThinU | Eigen::ComputeThinV);
	int rows_V_left = svd_left.matrixV().rows();
	int cols_V_left = svd_left.matrixV().cols();
	assert( cols_A == rows_V_left && cols_A == cols_V_left );
	Eigen::MatrixXf matrixV_left(rows_V_left, cols_V_left);
	matrixV_left = svd_left.matrixV();
	x = matrixV_left.col(cols_V_left-1);
	// set plane parameters
	param_left_plane << x(1), x(2), x(3), x(0);
	d = std::sqrt(x(1)*x(1) + x(2)*x(2) + x(3)*x(3));
	param_left_plane = param_left_plane/d;
	
	// back plane
	int rows_A_back = 0;
	// front plane
	for (int i = 0; i < cloud_back_plane.size(); ++i)
		rows_A_back += cloud_back_plane[i]->points.size();
	Eigen::MatrixXf A_back(rows_A_back, cols_A); 
	A_back.setZero();
	for (int j = 0; j < cloud_back_plane.size(); ++j)
	{
		for ( size_t i = 0; i < cloud_back_plane[j]->points.size(); ++i )
		{
			A_back(i, 0) = 1;
			A_back(i, 1) = cloud_back_plane[j]->points[i].x;
			A_back(i, 2) = cloud_back_plane[j]->points[i].y;
			A_back(i, 3) = cloud_back_plane[j]->points[i].z;
		}
	}
	// solve the least-squares problem using SVD
	Eigen::JacobiSVD<Eigen::MatrixXf> svd_back(A_back.transpose()*A_back, Eigen::ComputeThinU | Eigen::ComputeThinV);
	int rows_V_back = svd_back.matrixV().rows();
	int cols_V_back = svd_back.matrixV().cols();
	assert( cols_A == rows_V_back && cols_A == cols_V_back );
	Eigen::MatrixXf matrixV_back(rows_V_back, cols_V_back);
	matrixV_back = svd_back.matrixV();
	x = matrixV_back.col(cols_V_back-1);
	// set plane parameters
	param_back_plane << x(1), x(2), x(3), x(0);
	d = std::sqrt(x(1)*x(1) + x(2)*x(2) + x(3)*x(3));
	param_back_plane = param_back_plane/d;

	// right plane
	int rows_A_right = 0;
	// front plane
	for (int i = 0; i < cloud_right_plane.size(); ++i)
		rows_A_right += cloud_right_plane[i]->points.size();
	Eigen::MatrixXf A_right(rows_A_right, cols_A); 
	A_right.setZero();
	for (int j = 0; j < cloud_right_plane.size(); ++j)
	{
		for ( size_t i = 0; i < cloud_right_plane[j]->points.size(); ++i )
		{
			A_right(i, 0) = 1;
			A_right(i, 1) = cloud_right_plane[j]->points[i].x;
			A_right(i, 2) = cloud_right_plane[j]->points[i].y;
			A_right(i, 3) = cloud_right_plane[j]->points[i].z;
		}
	}
	// solve the least-squares problem using SVD
	Eigen::JacobiSVD<Eigen::MatrixXf> svd_right(A_right.transpose()*A_right, Eigen::ComputeThinU | Eigen::ComputeThinV);
	int rows_V_right = svd_right.matrixV().rows();
	int cols_V_right = svd_right.matrixV().cols();
	assert( cols_A == rows_V_right && cols_A == cols_V_right );
	Eigen::MatrixXf matrixV_right(rows_V_right, cols_V_right);
	matrixV_right = svd_right.matrixV();
	x = matrixV_right.col(cols_V_right-1);
	// set plane parameters
	param_right_plane << x(1), x(2), x(3), x(0);
	d = std::sqrt(x(1)*x(1) + x(2)*x(2) + x(3)*x(3));
	param_right_plane = param_right_plane/d;
}

/**
 * @brief plane model fiting (reconstruction) using pcl function
 * plane in Hessian Normal form: p + a*x + b*y + c*z = 0, subject to a^2+b^2+c^2 = 1
 *
 * @param[in]: cloud_front_plane, cloud_left_plane, cloud_back_plane, cloud_right_plane
 * @param[out]: param_front_plane, param_left_plane, param_back_plane, param_right_plane
 */
void LaserOdometry::planeModelFittingPCL(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_front_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_left_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_back_plane,
			   const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_right_plane,
			   Eigen::Vector4f& param_front_plane, Eigen::Vector4f& param_left_plane,
			   Eigen::Vector4f& param_back_plane, Eigen::Vector4f& param_right_plane)
{
	// Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType (pcl::SACMODEL_PLANE);
//    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMethodType (pcl::SAC_RRANSAC);
    seg.setDistanceThreshold(0.01);

	//    ax + by + cz + h = 0
	//    a = coefficients->values[0], b = coefficients->values[1],
	//    c = coefficients->values[2], h = coefficients->values[3]
	// front plane
	// initialize PointClouds
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_fp (new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < cloud_front_plane.size(); ++i)
		*cloud_fp += *cloud_front_plane_[i];
	if ( cloud_fp->points.size() > 0)
	{
		pcl::ModelCoefficients::Ptr coefficients_fp (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers_fp (new pcl::PointIndices);
		// Mandatory
		seg.setInputCloud(cloud_fp);
		seg.segment(*inliers_fp, *coefficients_fp);
		if (4 == coefficients_fp -> values.size())
			param_front_plane << coefficients_fp -> values[0], coefficients_fp -> values[1],
					coefficients_fp -> values[2], coefficients_fp -> values[3];
	}         

        // left plane
        // initialize cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_lp (new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < cloud_left_plane.size(); ++i)
		*cloud_lp += *cloud_left_plane_[i];
	if ( cloud_lp->points.size() > 0 )
	{
		pcl::ModelCoefficients::Ptr coefficients_lp (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers_lp (new pcl::PointIndices);
		seg.setInputCloud(cloud_lp);
		seg.segment(*inliers_lp, *coefficients_lp);
		if (4 == coefficients_lp -> values.size())
			param_left_plane << coefficients_lp -> values[0], coefficients_lp -> values[1],
					coefficients_lp -> values[2], coefficients_lp -> values[3]; 
	} 

        // back plane
        // initialize cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_bp (new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < cloud_back_plane.size(); ++i)
		*cloud_bp += *cloud_back_plane_[i];
	if ( cloud_bp->points.size() > 0 )
	{
	        pcl::ModelCoefficients::Ptr coefficients_bp (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers_bp (new pcl::PointIndices);
		seg.setInputCloud(cloud_bp);
		seg.segment(*inliers_bp, *coefficients_bp);
		if (4 == coefficients_bp -> values.size())
			param_back_plane << coefficients_bp -> values[0], coefficients_bp -> values[1],
					coefficients_bp -> values[2], coefficients_bp -> values[3];  
	}

        // right plane
        // initialize cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rp (new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < cloud_right_plane.size(); ++i)
		*cloud_rp += *cloud_right_plane_[i];
	if ( cloud_rp->points.size() > 0 )
	{
	        pcl::ModelCoefficients::Ptr coefficients_rp (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers_rp (new pcl::PointIndices);
		seg.setInputCloud(cloud_rp);
		seg.segment(*inliers_rp, *coefficients_rp);
		if (4 == coefficients_rp -> values.size())
			param_right_plane << coefficients_rp -> values[0], coefficients_rp -> values[1],
					coefficients_rp -> values[2], coefficients_rp -> values[3]; 
	} 
}

void LaserOdometry::cloudComparationVisualization(const pcl::PointCloud<pcl::PointXYZ>::Ptr& scan_cloud_in,
						  const std::string& worldROS_frame,
						const double& x_geo, const double& y_geo, const double& z_geo,
						const double& roll_geo, const double& pitch_geo, const double& yaw_geo,
						const double& x_reg, const double& y_reg, const double& z_reg,
						const double& roll_reg, const double& pitch_reg, const double& yaw_reg,
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_geo,
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_reg)
{
	// define a transform
	Eigen::Affine3f wROS_T_laser = Eigen::Affine3f::Identity();
	Eigen::Matrix3f w_R_v, Rx, Ry, Rz, R_x_pi;
	R_x_pi = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX());
	// define a transfered cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>(scan_cloud_in->points.size(), 1));

	// generate geometry estimation cloud in the world frame
	// set a translation
    	wROS_T_laser.translation() << x_geo, y_geo, z_geo;
	// set the rotation matrix
	Rx = Eigen::AngleAxisf(roll_geo, Eigen::Vector3f::UnitX());
	Ry = Eigen::AngleAxisf(pitch_geo, Eigen::Vector3f::UnitY());
	Rz = Eigen::AngleAxisf(yaw_geo, Eigen::Vector3f::UnitZ());
	w_R_v = R_x_pi*Rz * Ry * Rx*R_x_pi;
	wROS_T_laser.rotate(w_R_v);
	// apply transformation from world ROS (or defined inertial) to laser
	pcl::transformPointCloud(*scan_cloud_in, *scan_cloud_transformed, wROS_T_laser);
	// color the cloud
	pcl::PointXYZRGB pt;
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
	for (int i = 0; i < scan_cloud_transformed->points.size(); ++i)
	{
		pt.x = scan_cloud_transformed->points[i].x;
		pt.y = scan_cloud_transformed->points[i].y;
		pt.z = scan_cloud_transformed->points[i].z;
		cloud_geo->push_back(pt);
	}
 	// set the message header
        cloud_geo->header.frame_id = worldROS_frame;
	cloud_geo->header.stamp = scan_cloud_in->header.stamp;


	// generate registration based estimation in the world frame
	// set a translation
    	wROS_T_laser.translation() << x_reg, y_reg, z_reg;
	// set the rotation matrix
	Rx = Eigen::AngleAxisf(roll_reg, Eigen::Vector3f::UnitX());
	Ry = Eigen::AngleAxisf(pitch_reg, Eigen::Vector3f::UnitY());
	Rz = Eigen::AngleAxisf(yaw_reg, Eigen::Vector3f::UnitZ());
	w_R_v = R_x_pi*Rz * Ry * Rx*R_x_pi;
	wROS_T_laser.rotate(w_R_v);
	// apply transformation from world ROS (or defined inertial) to laser
	pcl::transformPointCloud(*scan_cloud_in, *scan_cloud_transformed, wROS_T_laser);
	// color the cloud
        pt.r = 0;
        pt.g = 0;
        pt.b = 255;
	for (int i = 0; i < scan_cloud_transformed->points.size(); ++i)
	{
		pt.x = scan_cloud_transformed->points[i].x;
		pt.y = scan_cloud_transformed->points[i].y;
		pt.z = scan_cloud_transformed->points[i].z;
		cloud_reg->push_back(pt);
	}
 	// set the message header
        cloud_reg->header = cloud_geo->header;
}

void LaserOdometry::planeVisualization(const Eigen::Vector4f& param_front_plane, const Eigen::Vector4f& param_left_plane,
                 const Eigen::Vector4f& param_back_plane, const Eigen::Vector4f& param_right_plane,
                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr& front_plane, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& left_plane,
                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr& back_plane, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& right_plane)
{
    int n = 200;
    float p_length = 1.5;
    float v_length = 10;
    pcl::PointXYZRGB pt;
    // front line
    pt.r = 255;
    pt.g = 0;
    pt.b = 0;
    for ( int i = 0; i < n; ++i )
    {
        pt.y = -p_length+ i*(2*p_length/n);
        for ( int j = 0; j < n; ++j )
        {
            pt.z = 0 + j*(v_length/n);
            pt.x = -( param_front_plane(1)*pt.y + param_front_plane(2)*pt.z + param_front_plane(3))/param_front_plane(0);
            front_plane->push_back(pt);
        }
    }
    // left line
    pt.r = 0;
    pt.g = 255;
    pt.b = 0;
    for ( int i = 0; i < n; ++i )
    {
        pt.x = -p_length + i*(2*p_length/n);
        for ( int j = 0; j < n; ++j )
        {
            pt.z = 0 + j*(v_length/n);
            pt.y = -( param_left_plane(0)*pt.x + param_left_plane(2)*pt.z + param_left_plane(3))/param_left_plane(1);
            left_plane->push_back(pt);
        }
    }
    // back line
    pt.r = 0;
    pt.g = 0;
    pt.b = 255;
    for ( int i = 0; i < n; ++i )
    {
        pt.y = -p_length + i*(2*p_length/n);
        for ( int j = 0; j < n; ++j )
        {
            pt.z = 0 + j*(v_length/n);
            pt.x = -( param_back_plane(1)*pt.y + param_back_plane(2)*pt.z + param_back_plane(3))/param_back_plane(0);
            back_plane->push_back(pt);
        }
    }
    // right line
    pt.r = 255;
    pt.g = 126;
    pt.b = 0;
    for ( int i = 0; i < n; ++i )
    {
        pt.x = -p_length + i*(2*p_length/n);
        for ( int j = 0; j < n; ++j )
        {
            pt.z = 0 + j*(v_length/n);
            pt.y = -( param_right_plane(0)*pt.x + param_right_plane(2)*pt.z + param_right_plane(3))/param_right_plane(1);
            right_plane->push_back(pt);
        }
    }
}

/**
 * @brief descripe the variables
 * scan_cloud_in: scan_in in PCL format
 * line_list1: preliminary line segment list with discarding short line segments
 * line_list2: selecting the nearest front line segment as the graph structure base
 * line_list3: the line segments satisfying the orientation constraints (version one)
 * line_list4: the line segments satisfying the structured graph (version two)
 */
void LaserOdometry::scanCallback(
        const sensor_msgs::LaserScan::ConstPtr& scan_in)
{
//    ROS_INFO_STREAM("Receiving and processing scan data ...");

    if (!b_system_inited_) {
        init_time_ = scan_in->header.stamp.toSec();
        b_system_inited_ = true;
    }

    ++scan_counter_;
    ROS_INFO_STREAM("scan counter " << scan_counter_ );

    time_scan_last_ = time_scan_cur_;
    time_scan_cur_ = scan_in->header.stamp.toSec();
    time_lasted_ = time_scan_cur_ - init_time_;

    scan_time_.push_back(time_scan_cur_);

    // reset the state
    b_geometry_estimation_ = false;
    b_geometry_estimation_ok_ = false;
    b_registration_estimation_ = false;
    b_registration_estimation_ok_ = false;

    bool b_debug = false;

    // visualization
    bool b_cloud_in_pub = false;
    bool b_convexHull_pub = false;
    bool b_line_list_pub = false;
    bool b_object_features_pub = false;
    bool b_registrated_cloud_pub = false;

    // Convert sensor_msgs::LaserScan to pcl point cloud
    int cloud_size = scan_in->ranges.size();
    pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud_in(new pcl::PointCloud<pcl::PointXYZ>(cloud_size, 1));
    float range_min = 0.5;
    float range_max = 30;
    sensorMsgsToPCLCloud( scan_in, range_min, range_max, scan_cloud_in );
    b_cloud_in_pub = true;

    // object boundary convex hull extraction
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_convex_hull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_convex_hull2 (new pcl::PointCloud<pcl::PointXYZ>);
    convexHull(scan_cloud_in, cloud_convex_hull, cloud_convex_hull2);

    // visualize convex hull
    geometry_msgs::PolygonStamped convex_polygon_points;
//    convexHullVisualization(scan_in, cloud_convex_hull, convex_polygon_points);
    convexHullVisualization(scan_in, cloud_convex_hull2, convex_polygon_points);
    b_convexHull_pub = true;

    // Extracting line segments from the convex hull contour
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_list1 (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_list2 (new pcl::PointCloud<pcl::PointXYZ>);
    lineSegmentDetection(cloud_convex_hull2, line_list1, line_list2);

//    pcl::ProjectInliers<pcl::PointXYZ> proj;
//    proj.setModelType(pcl::SACMODEL_LINE);

    // line segment identification
    bool b_front_found = false;
    bool b_left_found = false;
    bool b_back_found = false;
    bool b_right_found = false;

    pcl::PointCloud<pcl::PointXYZ>::Ptr line_list3
            (new pcl::PointCloud<pcl::PointXYZ>(line_list2->points.size(), 1));
    pcl::PointCloud<pcl::PointXYZ>::Ptr line_list4
            (new pcl::PointCloud<pcl::PointXYZ>(line_list2->points.size(), 1));
    std::vector<int> graph_indices; // for line_list4, graph_indices.size() should be equal to j_temp*2.

    if ( b_debug )
        lineSegmentIdentification(delta_graph_, line_list2, line_list3,
                                   line_list4, graph_indices,
                                   b_front_found, b_left_found,
                                   b_back_found, b_right_found );


    pcl::PointCloud<pcl::PointXYZI>::Ptr line_list5
            ( new pcl::PointCloud<pcl::PointXYZI> (8,1) );
    lineSegmentIdentification2(delta_graph_, line_list2, line_list5,
                                   b_front_found, b_left_found,
                                   b_back_found, b_right_found );
    b_line_list_pub = true;

    if ( b_debug )
        deBugLineList(line_list4, line_list5,
                           b_front_found, b_left_found, b_back_found, b_right_found);

    if (b_front_found && b_left_found && b_back_found && b_right_found)
    {
        b_geometry_estimation_= true;
    }

    /**
     * \brief geometry feature detection
     *
     * \param[in]: scan_cloud_in, line_list5, b_front_found, b_left_found, b_back_found, b_right_found
     * \param[out]: geometry points and their states
     * \param[out]: four cloud for four boundary lines
     */
    Eigen::Vector4f pt_front_left(0, 0, 0, 0);
    Eigen::Vector4f pt_front_right(0, 0, 0, 0);
    Eigen::Vector4f pt_front_middle(0, 0, 0, 0);
    Eigen::Vector4f pt_left_back(0, 0, 0, 0);
    Eigen::Vector4f pt_right_back(0, 0, 0, 0);
    Eigen::Vector4f pt_center(0, 0, 0, 0);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_front_line(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_left_line(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_back_line(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_right_line(new pcl::PointCloud<pcl::PointXYZ>);
    // Visualize the geometry feature points of the object:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_feature_points(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZRGB>);

    if ( b_debug )
        geometryFeatureDetectionSimple(line_list5, b_front_found, b_left_found,
                                       b_back_found, b_right_found, pt_center);

	if (b_front_found && !b_geometry_estimation_ )	// calculate z, roll, pitch, yaw
	{
		assert( line_list5->points.size() >=2 );
		pt_front_left << line_list5->points[1].x, line_list5->points[1].y, 0, 1;
		pt_front_right << line_list5->points[0].x, line_list5->points[0].y, 0, 1;
		// or calculating front left and front right points using line fitting
	}	
	if ( b_geometry_estimation_ )
	{
	    // proposed geometry feature detection method
	    geometryFeatureDetection2( scan_cloud_in, line_list5, b_front_found, b_left_found,
		    b_back_found, b_right_found, pt_front_left, pt_front_right, pt_front_middle,
		    pt_left_back, pt_right_back, pt_center, cloud_front_line, cloud_left_line,
		    cloud_back_line, cloud_right_line );

	    geometryFeatureVisualization(pt_front_left, pt_front_right, pt_left_back, pt_right_back, pt_center,
					cloud_front_line, cloud_left_line, cloud_back_line, cloud_right_line,
			                 object_feature_points, cloud_clustered);

	    object_feature_points->header = line_list5->header;
	    cloud_clustered->header = line_list5->header;
	    b_object_features_pub = true;
	}

    /** Geometry Feature based Pose Estimation */
    // initialize geometry features and pose
    Eigen::Matrix3f w_R_v_geo;	// rotation of the vehicle frame with respect to the world frame
    w_R_v_geo.setZero();
    Eigen::Vector3f i_rot_imu_geo(0, 0, 0); 	// rotation vector of the  imu frame with respect to the inertial frame
    Eigen::Vector3f w_t_v_geo(0, 0, 0);	// translation of the vehicle frame with respect to the world frame

    Eigen::Matrix3f w_R_v_geoFl;	// rotation of the vehicle frame with respect to the world frame
    w_R_v_geoFl.setZero();
    Eigen::Vector3f i_rot_imu_geoFl(0, 0, 0); 	// rotation vector of the  imu frame with respect to the inertial frame
    Eigen::Vector3f w_t_v_geoFl(0, 0, 0);	// translation of the vehicle frame with respect to the world frame
    // laser to object for visualization
    Eigen::Matrix3f laser_R_object_geo;
    laser_R_object_geo.setZero();
    Eigen::Vector3f laser_t_object_geo(0, 0, 0);
    Eigen::Matrix3f laser_R_object_geoFl;
    laser_R_object_geoFl.setZero();
    Eigen::Vector3f laser_t_object_geoFl(0, 0, 0);

    bool b_rotation_ok = false;
    bool b_translation_ok = false;

    // pose estimation using geometry method
    Eigen::Vector3d imu_gamma;
    imu_gamma << gamma_0_[att_counter_], gamma_1_[att_counter_], gamma_2_[att_counter_];
    double laser_altimeter = altitude_[alt_counter_];
    if (b_geometry_estimation_)
    {
        geometryPoseEstimation( imu_gamma, laser_altimeter, pt_front_left, pt_front_right,
                pt_center, w_R_v_geo, i_rot_imu_geo, w_t_v_geo, laser_R_object_geo, laser_t_object_geo,
                laser_R_object_geoFl, laser_t_object_geoFl, b_rotation_ok, b_translation_ok);

        b_geometry_estimation_ok_ = true;
        x_geo_.push_back(w_t_v_geo[0]);
        y_geo_.push_back(w_t_v_geo[1]);
        z_geo_.push_back(w_t_v_geo[2]);
        roll_geo_.push_back(i_rot_imu_geo[0]);
        pitch_geo_.push_back(i_rot_imu_geo[1]);
        yaw_geo_.push_back(i_rot_imu_geo[2]);
    }
    else if ( b_front_found )
    {
        geometryPoseEstimation_frontLine( imu_gamma, laser_altimeter, pt_front_left, pt_front_right, i_rot_imu_geoFl, w_t_v_geoFl, laser_R_object_geoFl, laser_t_object_geoFl );
        x_geo_fl_.push_back(w_t_v_geoFl[0]);
        y_geo_fl_.push_back(w_t_v_geoFl[1]);
        z_geo_fl_.push_back(w_t_v_geoFl[2]);
        roll_geo_fl_.push_back(i_rot_imu_geoFl[0]);
        pitch_geo_fl_.push_back(i_rot_imu_geoFl[1]);
        yaw_geo_fl_.push_back(i_rot_imu_geoFl[2]);
    }

    /** pose estimation using single view registration method (xyzY) */
     /**
      * @brief 3D rigid Motion Estimation using LM
      *
      * \param[in] x1 The identifed cloud in laser frame
      * \param[in] x2 The model planes in inertial frame
      * \param[out] t The initial 3x1 translation
      * \param[out] R The initial 3x3 rotation
      *
      * @param[in] cloud_front_line, cloud_left_line, cloud_back_line, cloud_right_line in the laser frame
      * @param[in] w_t_v_last = [x(k), y(k), z(k)] - translation of laser with respect to the ros world frame
      * @param[in] i_rot_imu_yaw_last = [ roll(k+1), pitch(k+1), yaw(k) ] - rotation vector of imu with respect to inertial
      * \return none
      */
     // registration estimation
//     if (b_plane_model_ok_)
    if (b_plane_model_ok_ && b_geometry_estimation_)
    {
        // do registration estimation
        b_registration_estimation_ = true;
    }

     Eigen::Vector3d rot_lm(0, 0, 0);
     Eigen::Vector3d t_lm(0, 0, 0);
     double yaw_lm;
     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_geo(new pcl::PointCloud<pcl::PointXYZRGB>);
     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_reg(new pcl::PointCloud<pcl::PointXYZRGB>);
     if ( b_registration_estimation_ )
     {
        // the input for LMA
        Eigen::Vector3d t, r;
        assert(x_geo_.size() > 1);
        assert(y_geo_.size() > 1);
        assert(z_geo_.size() > 1);
        assert(roll_geo_.size() > 1);
        assert(pitch_geo_.size() > 1);
        assert(yaw_geo_.size() > 1);
        t << x_geo_.rbegin()[1], y_geo_.rbegin()[1], z_geo_.rbegin()[1];
        r << roll_geo_.rbegin()[0], pitch_geo_.rbegin()[0], yaw_geo_.rbegin()[1];

        int n_unknowns = cloud_front_line->points.size() + cloud_left_line->points.size()
            + cloud_back_line->points.size() + cloud_right_line->points.size();

        int info;

        lm_SingleViewEstimation_xyzY_functor functor( 4, n_unknowns,
                                cloud_front_line, cloud_left_line,
                                cloud_back_line, cloud_right_line,
                                            param_front_plane_, param_left_plane_,
                                            param_back_plane_, param_right_plane_,
                                            t, r );

         Eigen::NumericalDiff<lm_SingleViewEstimation_xyzY_functor> numDiff( functor );

         Eigen::LevenbergMarquardt<Eigen::NumericalDiff<lm_SingleViewEstimation_xyzY_functor> >
                 lm( numDiff );
         lm.parameters.maxfev = 1000;
         Eigen::VectorXd xlm;
         xlm.setZero(4);

         info = lm.minimize( xlm );

         // update the refined results
         const Eigen::Vector3d transAdd = xlm.block<3, 1>( 0, 0 );
         const double yawAdd = xlm(3);
//        const Eigen::Matrix3d Rcor =
//        ( Eigen::AngleAxis<double>( rot( yawAdd ), Vec3::UnitY() ) ).toRotationMatrix();

         t_lm  = t + transAdd;
         yaw_lm = r[2] + yawAdd;
	 
         // check return values
        if ( 1 == info )
        {
            b_registration_estimation_ok_ = true;
//            x_singleReg_.push_back(t_lm(0));
//            y_singleReg_.push_back(t_lm(1));
//            z_singleReg_.push_back(t_lm(2));
//            roll_singleReg_.push_back(roll_geo_.back());
//            pitch_singleReg_.push_back(pitch_geo_.back());
//            yaw_singleReg_.push_back(yaw_lm);
        }
        x_singleReg_.push_back(t_lm(0));
        y_singleReg_.push_back(t_lm(1));
        z_singleReg_.push_back(t_lm(2));
        roll_singleReg_.push_back(roll_geo_.back());
        pitch_singleReg_.push_back(pitch_geo_.back());
        yaw_singleReg_.push_back(yaw_lm);

        // visualize the lm registration cloud vs geometry estimation cloud
        cloudComparationVisualization( scan_cloud_in, worldROS_frame_,
                        x_geo_.rbegin()[0], y_geo_.rbegin()[0], z_geo_.rbegin()[0],
                        roll_geo_.rbegin()[0], pitch_geo_.rbegin()[0], yaw_geo_.rbegin()[0],
                        x_singleReg_.rbegin()[0], y_singleReg_.rbegin()[0], z_singleReg_.rbegin()[0],
                        roll_singleReg_.rbegin()[0], pitch_singleReg_.rbegin()[0], yaw_singleReg_.rbegin()[0],
                        cloud_geo, cloud_reg);
     }

    if (b_geometry_estimation_ok_ && !b_registration_estimation_ok_)
        std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;

    /** pose estimation selection or fusion */
    Eigen::Matrix3f w_R_v;
    w_R_v.setZero();
    Eigen::Vector3f i_rot_imu(0, 0, 0);
    Eigen::Vector3f w_t_v(0, 0, 0);

    Eigen::Matrix3f laser_R_object;
    laser_R_object.setZero();
    Eigen::Vector3f laser_t_object(0, 0, 0);
    Eigen::Matrix3f laser_R_object_fl;
    laser_R_object_fl.setZero();
    Eigen::Vector3f laser_t_object_fl(0, 0, 0);

    if ( b_geometry_estimation_ok_ )
    {
    	w_R_v = w_R_v_geo;
        i_rot_imu = i_rot_imu_geo;
        w_t_v = w_t_v_geo;

        laser_R_object = laser_R_object_geo;
        laser_t_object = laser_t_object_geo;
        laser_R_object_fl = laser_R_object_geoFl;
        laser_t_object_fl = laser_t_object_geoFl;
    }
    else if ( b_front_found )
    {
        // set translation
        w_t_v(2) = w_t_v_geoFl(2);
        // set rotation
        i_rot_imu = i_rot_imu_geoFl;
        Eigen::Matrix3f Rx, Ry, Rz, R_x_pi, i_R_imu;
        R_x_pi = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX());
        Rx = Eigen::AngleAxisf(i_rot_imu(0), Eigen::Vector3f::UnitX());
        Ry = Eigen::AngleAxisf(i_rot_imu(1), Eigen::Vector3f::UnitY());
        Rz = Eigen::AngleAxisf(i_rot_imu(2), Eigen::Vector3f::UnitZ());
        i_R_imu = Rz * Ry * Rx;
        w_R_v = R_x_pi*i_R_imu*R_x_pi;

        // set laser to object
        laser_R_object_fl = laser_R_object_geoFl;
        laser_t_object_fl = laser_t_object_geoFl;
        // set the object frame to frame line center frame
        laser_R_object = laser_R_object_fl;
        laser_t_object = laser_t_object_fl;
    }
//    if ( b_registration_estimation_ )
//    {
//        // set translation
//        w_t_v << t_lm(0), t_lm(1), t_lm(2);
//        // set rotation
//        i_rot_imu << rot_lm(0), rot_lm(1), rot_lm(2);
//        Eigen::Matrix3f Rx, Ry, Rz, R_x_pi, i_R_imu;
//        R_x_pi = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX());
//        Rx = Eigen::AngleAxisf(i_rot_imu(0), Eigen::Vector3f::UnitX());
//        Ry = Eigen::AngleAxisf(i_rot_imu(1), Eigen::Vector3f::UnitY());
//        Rz = Eigen::AngleAxisf(i_rot_imu(2), Eigen::Vector3f::UnitZ());
//        i_R_imu = Rz * Ry * Rx;
//        w_R_v = R_x_pi*i_R_imu*R_x_pi;
//    }

    /** register the current laser scan to world frame (ROS world frame) */
    cloudRegistration( scan_cloud_in, w_R_v, w_t_v, pt_center, worldROS_frame_,
		       cloud_front_line, cloud_left_line, cloud_back_line, cloud_right_line,
                       cloud_worldROS_all_, cloud_worldROS_, cloud_worldROS_fail_,
		       cloud_front_plane_, cloud_left_plane_, cloud_back_plane_, cloud_right_plane_);

    // visualize the registration
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_planes(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloudPlaneVisualization(cloud_front_plane_, cloud_left_plane_, cloud_back_plane_, cloud_right_plane_, cloud_planes);
    // set the message header
    cloud_planes->header.frame_id = worldROS_frame_;
    cloud_planes->header.stamp = scan_cloud_in->header.stamp;

    b_registrated_cloud_pub = true;

    /** object planar modeling fitting (reconstruction) */
    if ( cloud_front_plane_.size() > 100 )
        b_plane_model_ok_ = true;
//    planeModelFitting( cloud_front_plane_, cloud_left_plane_, cloud_back_plane_, cloud_right_plane_,
//			param_front_plane_, param_left_plane_, param_back_plane_, param_right_plane_);

//	planeModelFittingTest( cloud_front_plane_, cloud_left_plane_, cloud_back_plane_, cloud_right_plane_,
//			param_front_plane_, param_left_plane_, param_back_plane_, param_right_plane_);
//    std::cout << "individual front plane: " << param_front_plane_ << std::endl;
//    std::cout << "individual left plane: " << param_left_plane_ << std::endl;
//    std::cout << "individual back plane: " << param_back_plane_ << std::endl;
//    std::cout << "individual right plane: " << param_right_plane_ << std::endl;

    planeModelFittingPCL( cloud_front_plane_, cloud_left_plane_, cloud_back_plane_, cloud_right_plane_,
			param_front_plane_, param_left_plane_, param_back_plane_, param_right_plane_);
//    std::cout << "pcl front plane: " << param_front_plane_ << std::endl;
//    std::cout << "pcl left plane: " << param_left_plane_ << std::endl;
//    std::cout << "pcl back plane: " << param_back_plane_ << std::endl;
//    std::cout << "pcl right plane: " << param_right_plane_ << std::endl;

    // visualize the fitted planes
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr front_plane(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr left_plane(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr back_plane(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr right_plane(new pcl::PointCloud<pcl::PointXYZRGB>);
    planeVisualization( param_front_plane_, param_left_plane_, param_back_plane_, param_right_plane_,
            front_plane, left_plane, back_plane, right_plane);
    front_plane->header = cloud_planes->header;
    left_plane->header = cloud_planes->header;
    back_plane->header = cloud_planes->header;
    right_plane->header = cloud_planes->header;

// -------------------------------- to be deleted --------------------------//
//    /** 3-2-4
//     * @brief motion smooth detection, if current motion is smooth, record the current data
//     */
//    // motion prediction
//    // expect v_(t-1) = v_t
//    bool b_smooth = true;
//    if (scan_counter_ > 2)
//    {
//        double dx = worldROS_x_laser_[scan_counter_] - worldROS_x_laser_[scan_counter_-1];
//        double dy = worldROS_y_laser_[scan_counter_] - worldROS_y_laser_[scan_counter_-1];
//        double dx_m1 = worldROS_x_laser_[scan_counter_-1] - worldROS_x_laser_[scan_counter_-2];
//        double dy_m1 = worldROS_y_laser_[scan_counter_-1] - worldROS_y_laser_[scan_counter_-2];
//        double dt = scan_time_[scan_counter_] - scan_time_[scan_counter_-1];
//        double dt_m1 = scan_time_[scan_counter_-1] - scan_time_[scan_counter_-2];
//        double dx_e = dx_m1*dt/dt_m1;
//        double dy_e = dy_m1*dt/dt_m1;
//        double x_e = worldROS_x_laser_[scan_counter_-1] + dx_e;
//        double y_e = worldROS_y_laser_[scan_counter_-1] + dy_e;
//        if (std::abs(x_e - wROS_x_laser)>0.3 || std::abs(y_e - wROS_y_laser)>0.3 )
//            b_smooth = false;
////        double k_LPF = 0.1;
////        lowPassFilter(k_LPF, x_e, );
////        motionControl::lowPassFilter(k_LPF_vl, expectedVl, _newVl);
//    }
//    else
//    {
//        worldROS_x_laser_f_.push_back(0);
//        worldROS_y_laser_f_.push_back(0);
//        worldROS_z_laser_f_.push_back(0);
//    }

//    /** buid the current safe pool (before the multi-view registration)
//     *  multi scans: cloud_multi_
//     *  multi delta t: delta_t_multi_
//     */
//    // generate delta_t and multi-view scans
//    int i_multi = scan_counter_ % num_multiView_;
//    // generate multi-view scans
//    cloud_multi_[i_multi] = cloud_clustered_refined;
//    // generate delta_t
//    if ( scan_counter_ > num_multiView_ )
//    {
//        delta_t_multi_[i_multi] = scan_time_[scan_counter_] - scan_time_[scan_counter_-1];
//    }
// -------------------------------- to be deleted --------------------------//

//    /** 3-2-6
//     * @brief select good initial motion esitmation
//     * switch between different estimators
//     */
//    // good initial estimation
//    bool b_good_geometry_esitmation = false;
//    bool b_recovery_by_registration = false;
//    if ( b_front_end && b_center_point && b_smooth
//         && b_front_found && b_left_found
//         && b_back_found && b_right_found
//         ) // the condition of geometry based method
//    {
//        b_good_geometry_esitmation = true;
//        w_x_l_.push_back(wROS_x_laser);
//        w_y_l_.push_back(wROS_y_laser);
//        w_z_l_.push_back(wROS_z_laser);
//        i_roll_imu_.push_back(roll);
//        i_pitch_imu_.push_back(pitch);
//        i_yaw_imu_.push_back(yaw);
//    }
//    else if ( b_plane_model_ok_ )  // use registration based method to recovery
//    {
//        b_recovery_by_registration = true;

//        // build the recovery function
//        // using incremental registration

//        /** the input for LMA
//        * @param n_unknowns: the unknown parameters
//        * @param Rotation vector: imu with respect to inertial frame
//        *       r = ( roll(scan_counter_ - num_multiView_),
//        *               pitch(scan_counter_ - num_multiView_),
//        *               yaw(scan_counter_ - num_multiView_) )
//        * @param Translation: laser with respect to the ros world frame
//        *       t = ( x(scan_counter_ - num_multiView_),
//        *               y(scan_counter_ - num_multiView_),
//        *               z(scan_counter_ - num_multiView_) )
//        */
//        int n_unknowns = 0;
//        assert( w_x_l_all_.size() > num_multiView_ );
//        for ( int i = 0; i < cloud_multi_.size(); i++ )
//        {
////            n_unknowns += cloud_multi_[i]->points.size();
//            n_unknowns += cloud_multi_.at(i)->points.size();
//        }

//        Eigen::Vector3d t, r;
//        assert( w_x_l_all_.size() > num_multiView_ );
//        assert( w_y_l_all_.size() > num_multiView_ );
//        assert( w_z_l_all_.size() > num_multiView_ );

//        t << w_x_l_all_.at( w_x_l_all_.size() - (num_multiView_ - 1) ) ,
//                w_y_l_all_.at( w_y_l_all_.size() - (num_multiView_ - 1) ),
//                w_z_l_all_.at( w_z_l_all_.size() - (num_multiView_ - 1) ) ;
//        r << i_roll_imu_all_.at( i_roll_imu_all_.size() - (num_multiView_ - 1) ),
//             i_pitch_imu_all_.at( i_roll_imu_all_.size() - (num_multiView_ - 1) ),
//             i_yaw_imu_all_.at( i_roll_imu_all_.size() - (num_multiView_ - 1) );

//        /** Multi-view registration optimization
//        */
//        lm_MultiViewEstimation_functor functor_multi(6, n_unknowns,
//                                                     cloud_multi_,
//                                                     delta_t_multi_,
//                                                     i_multi,
//                                                     coefficients_fp_,
//                                                     coefficients_lp_,
//                                                     coefficients_bp_,
//                                                     coefficients_rp_,
//                                                     t, r);
//        Eigen::NumericalDiff<lm_MultiViewEstimation_functor> numDiff_multi( functor_multi );

//        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<lm_MultiViewEstimation_functor> >
//                lm_multi(numDiff_multi);

//        lm_multi.parameters.maxfev = 1000;
//        Eigen::VectorXd xlm_multi;
//        xlm_multi.setZero(6);

//        lm_multi.minimize( xlm_multi );

//        // update the refined results
//        const Eigen::Vector3d transAdd = xlm_multi.block<3, 1>( 0, 0 );
//        const Eigen::Vector3d rotAdd = xlm_multi.block<3, 1>( 3, 0 );

//        wROS_x_laser = t[0] + num_multiView_ * transAdd[0];
//        wROS_y_laser = t[1] + num_multiView_ * transAdd[1];
//        wROS_z_laser = t[2] + num_multiView_ * transAdd[2];
//        roll = r[0] + num_multiView_ * rotAdd[0];
//        pitch = r[1] + num_multiView_ * rotAdd[1];
//        yaw = r[2] + num_multiView_ * rotAdd[2];
//        ROS_INFO_STREAM(" ******************* multi-view ");
//    }

//    // all the estimation
//    w_x_l_all_.push_back(wROS_x_laser);
//    w_y_l_all_.push_back(wROS_y_laser);
//    w_z_l_all_.push_back(wROS_z_laser);
//    i_roll_imu_all_.push_back(roll);
//    i_pitch_imu_all_.push_back(pitch);
//    i_yaw_imu_all_.push_back(yaw);

//    // update the environment model

//    /** 4
//     * @brief Registration: refinement step
//     */

//    /** 4-1
//     * @brief manage (record) registration data: obtain a sequence of scans
//     * Input: cluster_inliers_clouds from 2-5-2
//     */
//    bool b_left_record = false;
//    bool b_back_record = false;
//    bool b_right_record = false;
//    bool b_front_record = false;
//    if (b_good_geometry_esitmation)
//    {
//        if ( cluster_inliers_clouds.size() == graph_indices.size() )
//        {
//            scan_bar_counter_++;
//            int j_accem = scan_bar_counter_%n_assem_;
//            for (int i = 0; i < cluster_inliers_clouds.size(); i++ )
//            {
//                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
//                pcl::transformPointCloud(*cluster_inliers_clouds[i], *cloud_transformed, wROS_T_laser);
//                switch (graph_indices[i])
//                {
//                 // front
//                 case 0:
//                    if (!b_front_record)
//                    {
//                        if( scan_bar_counter_ > n_assem_-1)
//                        {
//                            cloud_front_bar_[j_accem]->clear();
//                            cloud_front_bar_[j_accem] = cloud_transformed;
//                        }
//                        else
//                        {
//                            cloud_front_bar_.push_back(cloud_transformed);
//                        }
//                        b_front_record = true;
//                    }
//                     break;
//                 // left
//                 case 1:
//                    if (!b_left_record)
//                    {
//                        if( scan_bar_counter_ > n_assem_-1)
//                        {
//                            cloud_left_bar_[j_accem]->clear();
//                            cloud_left_bar_[j_accem] = cloud_transformed;
//                        }
//                        else
//                        {
//                            cloud_left_bar_.push_back(cloud_transformed);
//                        }
//                        b_left_record = true;
//                    }
//                     break;
//                 // back
//                 case 2:
//                    if (!b_back_record)
//                    {
//                        if( scan_bar_counter_ > n_assem_-1)
//                        {
//                            cloud_back_bar_[j_accem]->clear();
//                            cloud_back_bar_[j_accem] = cloud_transformed;
//                        }
//                        else
//                        {
//                            cloud_back_bar_.push_back(cloud_transformed);
//                        }
//                        b_back_record = true;
//                    }
//                     break;
//                 // right
//                 case 3:
//                    if (!b_right_record)
//                    {
//                        if( scan_bar_counter_ > n_assem_-1)
//                        {
//                            cloud_right_bar_[j_accem]->clear();
//                            cloud_right_bar_[j_accem] = cloud_transformed;
//                        }
//                        else
//                        {
//                            cloud_right_bar_.push_back(cloud_transformed);
//                        }
//                        b_right_record = true;
//                    }
//                     break;
//                //                default: cout << "qwerty";
//                //                    break;
//                }
//            }
//            // compensation if any line segment is not detected
//            if (!b_front_record)
//            {
//                 pcl::PointCloud<pcl::PointXYZ>::Ptr pt(new pcl::PointCloud<pcl::PointXYZ>(1,1));
//                 pt->points[0].x = 0;
//                 pt->points[0].y = 0;
//                 pt->points[0].z = 0;
//                 if( scan_bar_counter_ > n_assem_-1)
//                 {
//                     cloud_front_bar_[j_accem]->clear();
//                     cloud_front_bar_[j_accem] = pt;
//                 }
//                 else
//                 {
//                    cloud_front_bar_.push_back(pt);
//                 }
//            }
//            if (!b_left_record)
//            {
//                 pcl::PointCloud<pcl::PointXYZ>::Ptr pt(new pcl::PointCloud<pcl::PointXYZ>(1,1));
//                 pt->points[0].x = 0;
//                 pt->points[0].y = 0;
//                 pt->points[0].z = 0;
//                 if( scan_bar_counter_ > n_assem_-1)
//                 {
//                     cloud_left_bar_[j_accem]->clear();
//                     cloud_left_bar_[j_accem] = pt;
//                 }
//                 else
//                 {
//                    cloud_left_bar_.push_back(pt);
//                 }
//            }
//            if (!b_back_record)
//            {
//                 pcl::PointCloud<pcl::PointXYZ>::Ptr pt(new pcl::PointCloud<pcl::PointXYZ>(1,1));
//                 pt->points[0].x = 0;
//                 pt->points[0].y = 0;
//                 pt->points[0].z = 0;
//                 if( scan_bar_counter_ > n_assem_-1)
//                 {
//                     cloud_back_bar_[j_accem]->clear();
//                     cloud_back_bar_[j_accem] = pt;
//                 }
//                 else
//                 {
//                    cloud_back_bar_.push_back(pt);
//                 }
//            }
//            if (!b_right_record)
//            {
//                 pcl::PointCloud<pcl::PointXYZ>::Ptr pt(new pcl::PointCloud<pcl::PointXYZ>(1,1));
//                 pt->points[0].x = 0;
//                 pt->points[0].y = 0;
//                 pt->points[0].z = 0;
//                 if( scan_bar_counter_ > n_assem_-1)
//                 {
//                     cloud_right_bar_[j_accem]->clear();
//                     cloud_right_bar_[j_accem] = pt;
//                 }
//                 else
//                 {
//                    cloud_right_bar_.push_back(pt);
//                 }
//            }
//        }
//        else
//            ROS_WARN_STREAM("The size of cluster_inliers_clouds is not equal to graph_indices ...");
//    }

//    /** 4-5
//     * @brief 3D rigid Motion Velocity Estimation using LM
//     * for multi-view registration, replacing 4-4
//     *
//     * \param[in] x1 The identifed cloud in laser frame
//     * \param[in] x2 The model planes in inertial frame
//     * \param[out] t The initial 3x1 translation
//     * \param[out] R The initial 3x3 rotation
//     *
//     * \return none
//     */
//     // the lines before are moved to recovery part

//    /** 5
//     * @brief fusion the transformation
//     */
//    // fusion
//    wROS_T_laser_last_ = wROS_T_laser;

//    // set the current motion estimation due to state
//     b_lm_motionEstimation = false; // if directly from geometry method
//    if ( b_lm_motionEstimation )
//    {
//        ROS_INFO_STREAM("LM estimation ...");
//        yaw = yaw_lm;
//        wROS_x_laser = t_lm[0];
//        wROS_y_laser = t_lm[1];
//        wROS_z_laser = t_lm[2];
//        // rotation matrix
//        R_x_pi = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX());
//        Rx = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
//        Ry = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
//        Rz = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
//        w_R_imu = Rz * Ry * Rx;
//        wROS_R_laser = R_x_pi*w_R_imu*R_x_pi;
//    }

//    /** 6: mapping
//     * @brief register the current scan to ROS world frame
//     * (or inertial frame) using the refined pose esitmation
//     *
//     * switch to use the estimator
//     */
//    // define a transform
//    Eigen::Affine3f wROS_T_laser_refine = Eigen::Affine3f::Identity();
//    if ( b_good_geometry_esitmation || b_recovery_by_registration )
//    {
//        pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>(scan_cloud_in->points.size(), 1));
//        // set a translation
//        wROS_T_laser_refine.translation() << wROS_x_laser, wROS_y_laser, wROS_z_laser;
//        // set the rotation matrix
//        wROS_T_laser_refine.rotate(wROS_R_laser);
//        // apply transformation from world ROS (or defined inertial) to laser
//        pcl::transformPointCloud(*scan_cloud_in, *scan_cloud_transformed, wROS_T_laser_refine);
//        *cloud_worldROS_refine_ += *scan_cloud_transformed;
//    }
//    // set the message header
//    pcl_conversions::toPCL(scan_in->header, cloud_worldROS_refine_->header);
//    cloud_worldROS_refine_->header.frame_id = worldROS_frame_;

//    /** 4-*
//     * @brief read the recorded registration data and visualize it
//     */
//    // the front line segment
//    if(b_front_end && b_center_point && b_smooth && cluster_inliers_clouds.size() == graph_indices.size())
//        cloud_bar_->clear();
//    for (int i=0; i < cloud_front_bar_.size(); i++)
//    {
//        if ( cloud_front_bar_[i]->points.size() > 1 )
//        {
//            for(int j = 0; j < cloud_front_bar_[i]->points.size(); j++)
//            {
//                pcl::PointXYZRGB p;
//                p.x = cloud_front_bar_[i]->points[j].x;
//                p.y = cloud_front_bar_[i]->points[j].y;
//                p.z = cloud_front_bar_[i]->points[j].z;
//                p.r = 255;
//                p.g = 255;
//                p.b = 255;
//                cloud_bar_->push_back(p);
//            }
//        }
//    }
//    // the left line segment
//    for (int i = 0; i < cloud_left_bar_.size(); i++)
//    {
//        if ( cloud_left_bar_[i]->points.size() > 1 )
//        {
//            for(int j = 0; j < cloud_left_bar_[i]->points.size(); j++)
//            {
//                pcl::PointXYZRGB p;
//                p.x = cloud_left_bar_[i]->points[j].x;
//                p.y = cloud_left_bar_[i]->points[j].y;
//                p.z = cloud_left_bar_[i]->points[j].z;
//                p.r = 255;
//                p.g = 0;
//                p.b = 0;
//                cloud_bar_->push_back(p);
//            }
//        }
//    }
//    // the back line segment
//    for (int i = 0; i < cloud_back_bar_.size(); i++)
//    {
//        if ( cloud_back_bar_[i]->points.size() > 1 )
//        {
//            for(int j = 0; j < cloud_back_bar_[i]->points.size(); j++)
//            {
//                pcl::PointXYZRGB p;
//                p.x = cloud_back_bar_[i]->points[j].x;
//                p.y = cloud_back_bar_[i]->points[j].y;
//                p.z = cloud_back_bar_[i]->points[j].z;
//                p.r = 255;
//                p.g = 255;
//                p.b = 255;
//                cloud_bar_->push_back(p);
//            }
//        }
//    }
//    // the right line segment
//    for (int i = 0; i < cloud_right_bar_.size(); i++)
//    {
//        if ( cloud_right_bar_[i]->points.size() > 1 )
//        {
//            for(int j = 0; j < cloud_right_bar_[i]->points.size(); j++)
//            {
//                pcl::PointXYZRGB p;
//                p.x = cloud_right_bar_[i]->points[j].x;
//                p.y = cloud_right_bar_[i]->points[j].y;
//                p.z = cloud_right_bar_[i]->points[j].z;
//                p.r = 0;
//                p.g = 0;
//                p.b = 255;
//                cloud_bar_->push_back(p);
//            }
//        }
//    }
//    // set the message header
//    pcl_conversions::toPCL(scan_in->header, cloud_bar_->header);
//    cloud_bar_->header.frame_id = worldROS_frame_;

    /**
     * \brief publishing data or messages
     */
    if ( b_cloud_in_pub )
        cloud_in_pub_.publish(scan_cloud_in);

    if ( b_convexHull_pub )
    {
        convex_vertices_pub_.publish(cloud_convex_hull);
        convex_vertices_pub2_.publish(cloud_convex_hull2);
        convex_polygon_pub_.publish(convex_polygon_points);
    }

    if ( b_line_list_pub )
    {
        line_list_pub_.publish(line_list1);
        line_list_pub2_.publish(line_list2);
        if ( b_debug )
            line_list_pub4_.publish(line_list4);
        line_list_pub5_.publish(line_list5);
    }

    if ( b_object_features_pub )
        object_features_pub_.publish(object_feature_points);

    if ( b_registrated_cloud_pub )
    {
        cloud_world_all_pub_.publish( cloud_worldROS_all_ );
        cloud_world_pub_.publish(cloud_worldROS_);
	cloud_world_fail_pub_.publish(cloud_worldROS_fail_);
//        cloud_world_pub_.publish(cloud_worldROS_refine_);
	clusters_pub_.publish(cloud_clustered);
        cloud_bar_pub_.publish(cloud_planes);

        cloud_geo_pub_.publish(cloud_geo);
        cloud_reg_pub_.publish(cloud_reg);

    }

    if ( b_plane_model_ok_ )
    {
        front_plane_pub_.publish(front_plane);
        left_plane_pub_.publish(left_plane);
        back_plane_pub_.publish(back_plane);
        right_plane_pub_.publish(right_plane);
    }

//    bool b_publish = false;
//    if ( b_publish )
//    {
//    //    concave_polygon_pub_.publish(concave_polygon_points);
//        clusters_pub_.publish(cloud_clustered);
//    //    clusters_refined_pub_.publish(cloud_clustered_refined);
//        clusters_inliers_pub_.publish(cluster_inliers);
//    }

//    scan_cloud_in->clear();
//    cloud_convex_hull->clear();

    /**
     * \brief publishing transformation
     *       /> inertial
     * world
     *       \> laser -> imu
     *                \> cross center frame
     */
//    // world to inertial frame
//    Eigen::Matrix3f R_x_pi;
//    R_x_pi = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX());
//    Eigen::Quaternionf quat_x_pi(R_x_pi);
//    geometry_msgs::TransformStamped transform_worldROS_to_inertial;
//    transform_worldROS_to_inertial.header.stamp = scan_in->header.stamp;
//    transform_worldROS_to_inertial.header.frame_id = worldROS_frame_;
//    transform_worldROS_to_inertial.child_frame_id = "inertial";
//    transform_worldROS_to_inertial.transform.rotation.w = quat_x_pi.w();
//    transform_worldROS_to_inertial.transform.rotation.x = quat_x_pi.x();
//    transform_worldROS_to_inertial.transform.rotation.y = quat_x_pi.y();
//    transform_worldROS_to_inertial.transform.rotation.z = quat_x_pi.z();
//    transform_worldROS_to_inertial.transform.translation.x = 0;
//    transform_worldROS_to_inertial.transform.translation.y = 0;
//    transform_worldROS_to_inertial.transform.translation.z = 0;
//    tf_broadcaster_.sendTransform(transform_worldROS_to_inertial);

    // world to laser (or vehicle)
    geometry_msgs::TransformStamped transform_worldROS_to_laser;
    transform_worldROS_to_laser.header.stamp = scan_in->header.stamp;
    transform_worldROS_to_laser.header.frame_id = worldROS_frame_;
    transform_worldROS_to_laser.child_frame_id = laser_frame_;
    Eigen::Quaternionf wROS_quat_laser(w_R_v);
    transform_worldROS_to_laser.transform.rotation.w = wROS_quat_laser.w();
    transform_worldROS_to_laser.transform.rotation.x = wROS_quat_laser.x();
    transform_worldROS_to_laser.transform.rotation.y = wROS_quat_laser.y();
    transform_worldROS_to_laser.transform.rotation.z = wROS_quat_laser.z();
    transform_worldROS_to_laser.transform.translation.x = w_t_v[0];
    transform_worldROS_to_laser.transform.translation.y = w_t_v[1];
    transform_worldROS_to_laser.transform.translation.z = w_t_v[2];
    tf_broadcaster_.sendTransform(transform_worldROS_to_laser);

//    // laser to imu
//    geometry_msgs::TransformStamped transform_laser_to_imu;
//    transform_laser_to_imu.header.stamp = scan_in->header.stamp;
//    transform_laser_to_imu.header.frame_id = laser_frame_;
//    transform_laser_to_imu.child_frame_id = "imu";
//    transform_laser_to_imu.transform.rotation.w = quat_x_pi.w();
//    transform_laser_to_imu.transform.rotation.x = quat_x_pi.x();
//    transform_laser_to_imu.transform.rotation.y = quat_x_pi.y();
//    transform_laser_to_imu.transform.rotation.z = quat_x_pi.z();
//    transform_laser_to_imu.transform.translation.x = 0;
//    transform_laser_to_imu.transform.translation.y = 0;
//    transform_laser_to_imu.transform.translation.z = 0;
//    tf_broadcaster_.sendTransform(transform_laser_to_imu);

    // laser to cross center
    geometry_msgs::TransformStamped transform_laser_to_object;
    transform_laser_to_object.header.stamp = scan_in->header.stamp;
    transform_laser_to_object.header.frame_id = laser_frame_;
    transform_laser_to_object.child_frame_id = "object";
    Eigen::Quaternionf laser_quat_object(laser_R_object);
    transform_laser_to_object.transform.rotation.w = laser_quat_object.w();
    transform_laser_to_object.transform.rotation.x = laser_quat_object.x();
    transform_laser_to_object.transform.rotation.y = laser_quat_object.y();
    transform_laser_to_object.transform.rotation.z = laser_quat_object.z();
    transform_laser_to_object.transform.translation.x = laser_t_object[0];
    transform_laser_to_object.transform.translation.y = laser_t_object[1];
    transform_laser_to_object.transform.translation.z = laser_t_object[2];
    tf_broadcaster_.sendTransform(transform_laser_to_object);

    // laser to the front line center
    geometry_msgs::TransformStamped transform_laser_to_object_fl;
    transform_laser_to_object_fl.header.stamp = scan_in->header.stamp;
    transform_laser_to_object_fl.header.frame_id = laser_frame_;
    transform_laser_to_object_fl.child_frame_id = "object_fl";
    Eigen::Quaternionf laser_quat_object_fl(laser_R_object_fl);
    transform_laser_to_object_fl.transform.rotation.w = laser_quat_object_fl.w();
    transform_laser_to_object_fl.transform.rotation.x = laser_quat_object_fl.x();
    transform_laser_to_object_fl.transform.rotation.y = laser_quat_object_fl.y();
    transform_laser_to_object_fl.transform.rotation.z = laser_quat_object_fl.z();
    transform_laser_to_object_fl.transform.translation.x = laser_t_object_fl[0];
    transform_laser_to_object_fl.transform.translation.y = laser_t_object_fl[1];
    transform_laser_to_object_fl.transform.translation.z = laser_t_object_fl[2];
    tf_broadcaster_.sendTransform(transform_laser_to_object_fl);
}

void LaserOdometry::lowPassFilter(	const double k_LPF,			// the parameter of low pass filter
                                    const float expectedValue,		// the expected value
                                    float &filteredValue)			// the filtered value
{
    filteredValue = filteredValue + k_LPF * ( expectedValue - filteredValue );
}

int main(int argc, char** argv)
{
    ros::init (argc, argv, "laser_odometry");
    LaserOdometry laser_odometry(100); // 100 the length
    ros::spin();
    ROS_INFO_STREAM("Finishing the laser odometry node...");

//    laser_odometry.debugTest();

    return 0;
}

/**
 * @brief for test
 */
void LaserOdometry::debugTest()
{
    Eigen::Vector4f v4;
    Eigen::VectorXf v1(1); 
    v1 << 1;
    v4.setZero();
    Eigen::Vector3f v3(1,2,3);
    v4.block<3,1>(0,0) = v3;
//    v4.block<1,1>(3,0) = v1;
    v4.tail<1>() = v1;
//    v4[3] = 1;

    std::cout << "v4 = " << v4 << std::endl;

	  Eigen::Matrix2d mat;
  mat << 1, 2,
         3, 4;
  mat = mat*mat;
  std::cout << "Now mat is mat:\n" << mat << std::endl;

	Eigen::MatrixXf A(10,3);
	A << 1, 1, 0.2,
		1, 2, 1.0,
		1, 3, 2.6,
		1, 4, 3.6,
		1, 5, 4.9,
		1, 6, 5.3,
		1, 7, 6.5,
		1, 8, 7.8,
		1, 9, 8.0,
		1, 10, 9.0;
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A.transpose()*A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	std::cout << "Its singular values are: " << svd.singularValues() << std::endl;
	std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;

	int rows = svd.matrixV().rows();
	int cols = svd.matrixV().cols();
	std::cout << "matrixV rows: " << rows << "matrixV cols: " << cols <<  std::endl;
	assert( 3 == rows && 3 == cols );
	Eigen::MatrixXf matrixV(rows, cols);
	matrixV = svd.matrixV();
	Eigen::VectorXf x( rows );
	x = matrixV.col(cols-1);
	std::cout << "front line parameters: " << x << std::endl;

//    Eigen::Vector3d transAdd = x.block<3, 1>( 0, 0 );
//    Eigen::Vector3d rotAdd = x.block<3, 1>( 3, 0 );

    //    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>);
    //    cloud1->width = 128;
    //    cloud1->height = 1;
    //    cloud1->points.resize (cloud1->width * cloud1->height);
    //    cloud1->points[0].x = 1;
    //    cloud1->points[0].y = 2;
    //    cloud1->points[0].z = 3;
    //    ROS_INFO_STREAM("cloud1 : ");
    //    ROS_INFO_STREAM("cloud1 width: " << cloud1->width);
    //    ROS_INFO_STREAM("cloud1 height: " << cloud1->height);
    //    ROS_INFO_STREAM("cloud1 size: " << cloud1->size() );
    //    ROS_INFO_STREAM("cloud1 points size: " << cloud1->points.size() );
    //    cloud1->points.resize (10);
    //    ROS_INFO_STREAM("cloud1 after points resize: ");
    //    ROS_INFO_STREAM("cloud1 width: " << cloud1->width);
    //    ROS_INFO_STREAM("cloud1 height: " << cloud1->height);
    //    ROS_INFO_STREAM("cloud1 size: " << cloud1->size() );
    //    ROS_INFO_STREAM("cloud1 points size: " << cloud1->points.size() );
    //    cloud1->resize(5);
    //    ROS_INFO_STREAM("cloud1 after resize: ");
    //    ROS_INFO_STREAM("cloud1 width: " << cloud1->width);
    //    ROS_INFO_STREAM("cloud1 height: " << cloud1->height);
    //    ROS_INFO_STREAM("cloud1 size: " << cloud1->size() );
    //    ROS_INFO_STREAM("cloud1 points size: " << cloud1->points.size() );

    //    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2
    //            (new pcl::PointCloud<pcl::PointXYZ>(128, 1));
    //    ROS_INFO_STREAM("cloud2 : ");
    //    ROS_INFO_STREAM("cloud2 width: " << cloud2->width);
    //    ROS_INFO_STREAM("cloud2 height: " << cloud2->height);
    //    ROS_INFO_STREAM("cloud2 size: " << cloud2->size() );
    //    ROS_INFO_STREAM("cloud2 points size: " << cloud2->points.size() );
    //    cloud2->points.resize (10);
    //    ROS_INFO_STREAM("cloud2 after points resize: ");
    //    ROS_INFO_STREAM("cloud2 width: " << cloud2->width);
    //    ROS_INFO_STREAM("cloud2 height: " << cloud2->height);
    //    ROS_INFO_STREAM("cloud2 size: " << cloud2->size() );
    //    ROS_INFO_STREAM("cloud2 points size: " << cloud2->points.size() );
    //    cloud2->resize(5);
    //    ROS_INFO_STREAM("cloud2 after resize: ");
    //    ROS_INFO_STREAM("cloud2 width: " << cloud2->width);
    //    ROS_INFO_STREAM("cloud2 height: " << cloud2->height);
    //    ROS_INFO_STREAM("cloud2 size: " << cloud2->size() );
    //    ROS_INFO_STREAM("cloud2 points size: " << cloud2->points.size() );

    //    for ( size_t i = 1; i < 2/2; ++i )
    //    {
    //        std::cout << "i: " << i << std::endl;
    //    }
    //    std::cout << "end loop... " << std::endl;
}

void LaserOdometry::deBugLineList(const pcl::PointCloud<pcl::PointXYZ>::Ptr& line_list4,
                   const pcl::PointCloud<pcl::PointXYZI>::Ptr& line_list5,
                   const bool& b_front_found,
                   const bool& b_left_found,
                   const bool& b_back_found,
                   const bool& b_right_found)
{
    ROS_INFO_STREAM("line list 4: " << line_list4->points.size());
    ROS_INFO_STREAM("line list 5: " << line_list5->points.size());
    int temp = -1;
    if ( b_front_found )
    {
        ++temp;
        ROS_INFO_STREAM("front line 4: " << "(" << line_list4->points[2*temp].x << ", " << line_list4->points[2*temp].y << ")"
                << " (" <<  line_list4->points[2*temp+1].x << ", " << line_list4->points[2*temp+1].y << ")");
    }
    if ( b_left_found )
    {
        ++temp;
        ROS_INFO_STREAM("left line 4: " << "(" << line_list4->points[2*temp].x << ", " << line_list4->points[2*temp].y << ")"
                << " (" <<  line_list4->points[2*temp+1].x << ", " << line_list4->points[2*temp+1].y << ")");
    }
    if ( b_back_found )
    {
        ++temp;
        ROS_INFO_STREAM("back line 4: " << "(" << line_list4->points[2*temp].x << ", " << line_list4->points[2*temp].y << ")"
                << " (" <<  line_list4->points[2*temp+1].x << ", " << line_list4->points[2*temp+1].y << ")");
    }
    if ( b_right_found )
    {
        ++temp;
        ROS_INFO_STREAM("right line 4: " << "(" << line_list4->points[2*temp].x << ", " << line_list4->points[2*temp].y << ")"
                << " (" <<  line_list4->points[2*temp+1].x << ", " << line_list4->points[2*temp+1].y << ")");
    }
    temp = -1;
    if ( b_front_found )
    {
        ++temp;
        ROS_INFO_STREAM("front line 5: " << "(" << line_list5->points[2*temp].x << ", " << line_list5->points[2*temp].y << ")"
                << " (" <<  line_list5->points[2*temp+1].x << ", " << line_list5->points[2*temp+1].y << ")");
    }
    if ( b_left_found )
    {
        ++temp;
        ROS_INFO_STREAM("left line 5: " << "(" << line_list5->points[2*temp].x << ", " << line_list5->points[2*temp].y << ")"
                << " (" <<  line_list5->points[2*temp+1].x << ", " << line_list5->points[2*temp+1].y << ")");
    }
    if ( b_back_found )
    {
        ++temp;
        ROS_INFO_STREAM("back line 5: " << "(" << line_list5->points[2*temp].x << ", " << line_list5->points[2*temp].y << ")"
                << " (" <<  line_list5->points[2*temp+1].x << ", " << line_list5->points[2*temp+1].y << ")");
    }
    if ( b_right_found )
    {
        ++temp;
        ROS_INFO_STREAM("right line 5: " << "(" << line_list5->points[2*temp].x << ", " << line_list5->points[2*temp].y << ")"
                << " (" <<  line_list5->points[2*temp+1].x << ", " << line_list5->points[2*temp+1].y << ")");
    }

    if (line_list5->size() != line_list4->size())
        ROS_INFO_STREAM("********************+++++++++++++++++++++ ");
}

void LaserOdometry::printEulerAngles()
{
    std::cout << "yaw:";
    for(int i =0; i < yaw_.size(); i++)
        std::cout << " " << yaw_[i];
    std::cout << "pitch:";
    for(int i =0; i < pitch_.size(); i++)
        std::cout << " " << pitch_[i];
    std::cout << "roll:";
    for(int i =0; i < roll_.size(); i++)
        std::cout << " " << roll_[i];
}
