function [joint_locations]= normalization_skel(tmpSkel,actionLen,n_joints)
% this function is to normalize the skeleton data
% Copyright (C) 2017 Haoyu Chen <chenhaoyucc@icloud.com>,
% center of Machine Vision and Signal Analysis,
% Department of Computer Science and Engineering,
% University of Oulu, Oulu, 90570, Finland

%getskeleton location normalized by hipcenter

jointskeletons = zeros(3,n_joints);
joint_locations = zeros(3,n_joints,actionLen);
for k = 1: actionLen
    skeletonloc = tmpSkel(k,:);
    %for joint = 1:20
    for joint = 1:12
        jointskeleton = skeletonloc(1 + (joint-1)*9: 3 + (joint-1)*9)';
        jointskeletons(:,joint) = jointskeleton;
    end
    joint_locations(:,:,k) = jointskeletons;
end