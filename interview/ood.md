// ================================================================= //
�Ķ���¼: 2013.10.11 
// ================================================================= //
http://www.cnblogs.com/whyandinside/archive/2012/10/26/2740612.html
OOD�����Ե�ʱҪ�ر�ע��Clarify the ambiguity���Ա��������Ƽȿ������������ֲ���over design��
��Ҫ�Լ����裬�ڿ���Ӧ��Ӧ��ʹ��ĳ�����ʱ�������Թ���ȷ�����������Ƿ����

OOD ���һ��ħ����ʵ��һ��rotation method

����OOD��ʵ����̵ȴ�ʱ��{
	class Cab{
		Capacity
		LoadWeight
		Floor
		Status
		serveRequest();
	}
	
	Request{
		floor, direction, weight;
	}
	
	class CenterControl{
		vector<Cab> cabs;
		AssignedCab(Request);
	}
}

Parking Lot{
	class car{
		int size, brand, model, color, etc.
	}
	
	class ParkingSpace{
		position, 
		car,
		enum status {vacant, notAvailable},
		subcalss handicap, compact, large etc
	}
	
	class OutletInLet{
		position
		status 
		subclass Entrance, Exit
	}
	
	class ParkingLot{
		vector<ParkingSpace> spaceArray;
		vector<Entrance> spaceArray;
		vector<Exit> spaceArray;
		
		int Size;
		int AvailableSpaceNum;
		HandleService(ParkingService){
		}
		
	}
	
	class ParkingService{
		status = {Rejected, Served, Parking, Entering, Exiting}
		car, 
		ParkingSpace,
		Extrance,
		Exit,
		EnteringTime, ExitingTime
		
	}

}

Restarent Reservation System{
	class Person{
		age, skins, 
		subclass Cook, Servant, Guest{Table, status}
	}
	
	class Service{
	}
	
	class Table{
		size, position, status
		vector<Service>
	}
	
	class Request{
		Person p; 
		subclass ServantRequest, GuestRequest, CookRequest
	}
	
	Restrant{
		vector<Table> tables;
		vector<Servant> servants;
		vector<Cook> cooks;
		vector<Material>
		
		HandleRequest(Person)
	}
}

http://www.cnblogs.com/quanfengnan/archive/2012/10/22/2733922.html

23.    ���һ����amazon��Ʒ��page,������������product info, reviews, recommendations (like ppl who buy this also buy),Ҫ��user customized.
24.    ���һ��youtube��type ahead search bar,�漰�����ݽṹ��distributed hashtable�͹����ڴ档